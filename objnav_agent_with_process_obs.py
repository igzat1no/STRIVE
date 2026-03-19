import os

import cv2
import habitat
import numpy as np
import open3d as o3d
import quaternion
import torch
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.maps import \
    colorize_draw_agent_and_fit_to_height
from loguru import logger

from cv_utils.gpt_utils import (ask_gpt_object_in_box,
                                check_again_object_in_bbox,
                                refine_tag_with_target_obj_list)
from cv_utils.stitch import combine_image, image_stitch_and_crop
from cv_utils.visualizer import visualize_mask
from mapper_with_process_obs import Instruct_Mapper
from mapping_utils.geometry import (gpu_cluster_filter, gpu_merge_pointcloud,
                                    gpu_pointcloud_from_array, pointcloud_distance,
                                    project_to_camera)
from mapping_utils.path_planning import path_planning
from mapping_utils.projection import (bresenham_3d, translate_grid_to_point,
                                      translate_point_to_grid,
                                      translate_single_point_to_grid)
from mapping_utils.transform import habitat_rotation


class HM3D_Objnav_Agent:

    def __init__(self,
                 env: habitat.Env,
                 mapper: Instruct_Mapper,
                 save_dir,
                 do_seg=True,
                 relocate=False,
                 gpt_relocate=True,
                 vlm='gemini'):
        self.env = env
        self.mapper = mapper
        self.episode_samples = 0
        self.planner = ShortestPathFollower(env.sim, 0.15, False)
        self.found_goal = False
        self.save_dir = save_dir
        self.do_seg = do_seg
        self.relocate = relocate
        self.gpt_relocate = gpt_relocate
        self.success_distance = 1.0
        self.stop_criterion = 0.7
        self.vlm = vlm

    def translate_objnav(self, object_goal):
        if object_goal.lower() == 'plant':
            return "Find the <%s>." % "potted_plant"
        # elif object_goal.lower() == "tv_monitor":
        #     return "Find the <%s>." % "television_set"
        else:
            return "Find the <%s>." % object_goal

    def reset_debug_probes(self):
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []
        self.segmentation_trajectory = []

        self.position_trajectory = []
        self.rotation_trajectory = []

        self.gpt_trajectory = []
        self.gptv_trajectory = []
        self.panoramic_trajectory = []

        self.obstacle_affordance_trajectory = []
        self.semantic_affordance_trajectory = []
        self.history_affordance_trajectory = []
        self.action_affordance_trajectory = []
        self.gpt4v_affordance_trajectory = []
        self.affordance_trajectory = []

        self.temporary_pcd = []
        self.temporary_depths = []
        self.angles = []
        self.mapper.current_obj_indices = []

    @property
    def position(self):
        return self.env.sim.get_agent_state().sensor_states['rgb'].position

    @property
    def rotation(self):
        return self.env.sim.get_agent_state().sensor_states['rgb'].rotation

    def reset(self, idx):
        self.episode_samples = idx + 1
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.mapper.initialize(self.position, self.rotation, self.env)
        self.instruct_goal = self.translate_objnav(self.env.current_episode.object_category)
        self.trajectory_summary = ""
        self.reset_debug_probes()

        self.best_distance = 2.0
        self.found_goal = False
        self.need_check_again = False
        self.just_come_back = False

        self.room_final = None
        self.waypoint = None
        self.on_node_flag = False
        self.travel_distance = 0.0
        self.start_end_episode_distance = self.env.get_metrics()['distance_to_goal']

        self.current_node_idx = 0
        self.update_trajectory()

    def rotate_panoramic(self, rotate_times=12):
        self.temporary_pcd = []
        temporary_images = []
        temporary_positions, temporary_rotations = [], []
        self.angles = []
        q_identity = quaternion.quaternion(1, 0, 0, 0)
        self.mapper.current_obj_indices = []

        self.B_classes, self.B_boxes, self.B_masks, self.B_confidences, self.B_visualization, \
            self.C_boxes, self.C_masks, self.C_confidences, self.C_visualization = \
                [], [], [], [], [], [], [], [], []

        for i in range(rotate_times):
            if self.env.episode_over:
                logger.info(f'Step: {self.env._elapsed_steps}')
                logger.info(f'Time: {self.env._elapsed_seconds}')
                return

            if self.mapper.current_navigable_pcd is None and self.mapper.current_pcd is None:
                temporary_pcd = gpu_pointcloud_from_array(np.zeros((0, 3)), np.zeros((0, 3)), self.mapper.pcd_device)
            else:
                temporary_pcd = gpu_merge_pointcloud(self.mapper.current_navigable_pcd,
                                                     self.mapper.current_pcd).voxel_down_sample(self.mapper.pcd_resolution)

            self.temporary_pcd.append(temporary_pcd)
            temporary_images.append(self.rgb_trajectory[-1])
            temporary_positions.append(self.mapper.current_position)
            temporary_rotations.append(self.mapper.current_rotation)

            self.angles.append(2 * np.arccos((q_identity.inverse() * self.rotation).w))
            self.obs = self.env.step(3)
            self.update_trajectory()

        temp_depths = self.temporary_depths[-13:-1]
        if self.episode_steps == 13:
            temporary_images[0] = self.rgb_trajectory[-1]
            temporary_positions[0] = self.mapper.current_position
            temporary_rotations[0] = self.mapper.current_rotation
            temp_depths[0] = self.temporary_depths[-1]

        if self.do_seg:
            self.rotate_segmentation(temporary_images, temp_depths,
                                     temporary_positions, temporary_rotations)

        self.just_come_back = False
        logger.info("object indices")
        logger.info(self.mapper.current_obj_indices)

    def rotate_segmentation(self, images, depths, positions, rotations):
        h, w, _ = images[0].shape
        C_objs = []
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}',
                    exist_ok=True)
        for i in range(12):
            prev = (i - 1) if i > 0 else 11
            nxt = (i + 1) if i < 11 else 0

            comb_img = combine_image(images[prev], images[i], images[nxt],
                                     self.mapper.camera_intrinsic)
            comb_depth = combine_image(depths[prev], depths[i], depths[nxt],
                                       self.mapper.camera_intrinsic)
            depth_vis = np.clip((comb_depth / 5.0 * 255.0), 0, 255).astype(np.uint8)
            B_classes, B_boxes, B_masks, B_confidences, \
                C_classes, C_boxes, C_masks, C_confidences = \
                    self.mapper.object_perceiver.perceive(
                        comb_img,
                        target=self.mapper.target,
                        target_list=self.mapper.target_list,
                        save_dir=self.save_dir,
                        episode_idx=self.episode_samples - 1,
                        episode_step=self.episode_steps,
                    )

            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/comb_img_{i}.jpg',
                comb_img)
            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/comb_depth_{i}.jpg',
                depth_vis)

            current_pos = [positions[prev], positions[i], positions[nxt]]
            current_rot = [rotations[prev], rotations[i], rotations[nxt]]
            depths_list = [depths[prev], depths[i], depths[nxt]]

            if not (B_boxes is None or B_boxes.shape[0] == 0):

                B_centers = (B_boxes[:, :2] + B_boxes[:, 2:]) * 0.5
                flag = (B_centers[:, 0] >= w) & (B_centers[:, 0] < 2 * w)
                B_classes = B_classes[flag.cpu().numpy()]
                B_centers = B_centers[flag]
                B_boxes = B_boxes[flag]
                B_masks = B_masks[flag]

                if not (B_boxes is None or B_boxes.shape[0] == 0):

                    B_confidences = B_confidences[flag]
                    B_visualization = visualize_mask(comb_img, B_boxes, B_confidences, B_classes, B_masks)

                    cv2.imwrite(
                        f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/B_dino_result_{i}.jpg',
                        B_visualization)

                    B_objs = self.mapper.get_object_entities_pano(comb_depth, comb_img,
                        current_pos, current_rot, B_classes, B_boxes, B_masks, B_confidences,
                        depths_list)

                    self.mapper.objects, obj_indices = self.mapper.associate_object_entities(
                        self.mapper.objects, B_objs)
                    self.mapper.current_obj_indices += obj_indices
                    self.mapper.object_pcd = self.mapper.update_object_pcd()

            if C_boxes is None or C_boxes.shape[0] == 0:
                continue

            C_centers = (C_boxes[:, :2] + C_boxes[:, 2:]) * 0.5
            flag = (C_centers[:, 0] >= w) & (C_centers[:, 0] < 2 * w)
            C_classes = C_classes[flag.cpu().numpy()]
            C_centers = C_centers[flag]
            C_boxes = C_boxes[flag]
            C_masks = C_masks[flag]

            if C_boxes is None or C_boxes.shape[0] == 0:
                continue

            C_confidences = C_confidences[flag]
            C_visualization = visualize_mask(comb_img, C_boxes, C_confidences, C_classes, C_masks)

            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_result_{i}.jpg',
                C_visualization)

            C_objs.append(self.mapper.get_object_entities_pano(comb_depth, comb_img,
                current_pos, current_rot, C_classes, C_boxes, C_masks, C_confidences,
                depths_list))

        # save pc
        obj_pcd = o3d.geometry.PointCloud()
        for i in range(len(C_objs)):
            for obj in C_objs[i]:
                points = obj.pcd_all.point.positions.cpu().numpy()
                colors = obj.pcd_all.point.colors.cpu().numpy()
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(points)
                new_pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_pcd_{i}.ply', new_pcd)
                obj_pcd = obj_pcd + new_pcd
        if len(obj_pcd.points) > 0:
            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_pcd.ply', obj_pcd)
        # original C objs (all)

        # combine the C_objs
        real_C_objs = []
        for i in range(len(C_objs)):
            for obj in C_objs[i]:
                overlap_score = []
                eval_pcd = obj.pcd
                eval_pcd_all = obj.pcd_all
                for prev_obj in real_C_objs:
                    prev_obj_pcd = prev_obj.pcd
                    prev_obj_pcd_all = prev_obj.pcd_all
                    cdist1 = pointcloud_distance(eval_pcd_all, prev_obj_pcd_all)
                    cdist2 = pointcloud_distance(prev_obj_pcd_all, eval_pcd_all)
                    cdist_all = torch.cat([cdist1, cdist2], dim=0)
                    overlap_condition = (cdist_all < 0.1)
                    overlap_condition1 = (cdist1 < 0.1)
                    overlap_condition2 = (cdist2 < 0.1)
                    overlap_score_tmp = (overlap_condition.sum() /
                                          (overlap_condition.shape[0] + 1e-6)).cpu().numpy().item()
                    overlap_score_tmp1 = (overlap_condition1.sum() /
                                            (overlap_condition1.shape[0] + 1e-6)).cpu().numpy().item()
                    overlap_score_tmp2 = (overlap_condition2.sum() /
                                            (overlap_condition2.shape[0] + 1e-6)).cpu().numpy().item()
                    if (overlap_score_tmp1 > 0.85 and overlap_score_tmp2 < 0.85) or (overlap_score_tmp1 < 0.85 and overlap_score_tmp2 > 0.85):
                        overlap_score_tmp = -1.0
                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! overlap score: {overlap_score_tmp}, {overlap_score_tmp1}, {overlap_score_tmp2}')
                    overlap_score.append(overlap_score_tmp)
                overlap_flag = [score > 0.25 for score in overlap_score]
                for j in range(len(overlap_flag)):
                    if overlap_flag[j]:
                        eval_pcd = gpu_merge_pointcloud(eval_pcd, real_C_objs[j].pcd)
                        eval_pcd_all = gpu_merge_pointcloud(eval_pcd_all, real_C_objs[j].pcd_all)
                        obj.confidence = max(obj.confidence, real_C_objs[j].confidence)
                obj.pcd = eval_pcd
                obj.pcd_all = eval_pcd_all

                need_to_move = []
                for j in range(len(overlap_flag)):
                    if overlap_flag[j]:
                        need_to_move.append(j)

                real_C_objs = [
                    obj_ for idx, obj_ in enumerate(real_C_objs) if idx not in need_to_move
                ]

                real_C_objs.append(obj)

        # save pc
        obj_pcd = o3d.geometry.PointCloud()
        iii = 0
        for obj in real_C_objs:
            obj.pcd = gpu_cluster_filter(obj.pcd)

            points = obj.pcd.point.positions.cpu().numpy()
            colors = obj.pcd.point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            obj_pcd = obj_pcd + new_pcd

            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_objs_{iii}.ply',
                new_pcd)
            iii += 1

        current_pos = self.mapper.current_position[:2]
        nw_ori = self.rotation
        nw_ori = habitat_rotation(nw_ori)
        nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
        for (i, obj) in enumerate(real_C_objs):
            obj.position = np.mean(obj.pcd.point.positions.cpu().numpy(), axis=0)
            center_pos = obj.position[:2]

            # calculate angle
            center_pos = center_pos - current_pos
            nw_ori = nw_ori / np.linalg.norm(nw_ori)
            center_pos = center_pos / np.linalg.norm(center_pos)
            dot_product = np.clip(np.dot(nw_ori, center_pos), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle = np.degrees(angle_rad)

            cross_product = np.cross(nw_ori, center_pos)
            if cross_product < 0:
                angle = 360 - angle

            angle = angle + 7.5
            if angle >= 360:
                angle -= 360
            image_ind = int(angle / 15)
            final_img = None
            final_box = None
            final_position = self.mapper.current_position
            final_rotation = None
            if image_ind % 2 == 0:
                image_ind = int(image_ind / 2)
                final_img = images[image_ind]
                final_rotation = rotations[image_ind]
            else:
                image_ind = int(image_ind / 2)
                nxt_ind = image_ind + 1
                if nxt_ind == 12:
                    nxt_ind = 0
                final_img = image_stitch_and_crop(images[image_ind],
                                                  images[nxt_ind],
                                                  self.mapper.camera_intrinsic)

                final_rotation = rotations[image_ind]
                deg = np.arccos(final_rotation[0][0])
                if final_rotation[0][2] > 0:
                    deg = np.pi * 2 - deg
                deg += (15 / 180) * np.pi
                final_rotation = np.array([[np.cos(deg), 0, -np.sin(deg)],
                                           [np.sin(deg), 0, np.cos(deg)], [0, 1, 0]])
            cv2.imwrite(
                f"{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_obj_image_{i}.jpg",
                final_img)

            camera_points = project_to_camera(obj.pcd_all, self.mapper.camera_intrinsic, final_position,
                                              final_rotation)
            camera_points = np.array(camera_points)
            camera_points = camera_points.T
            camera_points = np.array(camera_points[:, :2], dtype=np.int32)
            flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
                     (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
            camera_points = camera_points[flag]

            bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
            bbox[0] = np.maximum(bbox[0] - 3, 0)
            bbox[1] = np.minimum(bbox[1] + 3, [639, 479])
            final_box = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
            # save img
            img = final_img.copy()
            cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 255, 0), 2)
            cv2.imwrite(
                f"{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_obj_image_bbox_{i}.jpg",
                img)

            final_box = torch.tensor(final_box).unsqueeze(0)

            res = ask_gpt_object_in_box(final_img, final_box, self.save_dir, self.episode_samples-1, self.episode_steps, i, self.vlm)

            if res not in self.mapper.object_perceiver.classes:
                res = refine_tag_with_target_obj_list(res, self.mapper.target, self.save_dir, self.episode_samples-1, self.episode_steps, i, self.vlm)

            obj.num_list[res] = obj.num_list.pop(obj.tag)
            obj.conf_list[res] = obj.conf_list.pop(obj.tag)
            obj.tag = res

            if res == self.mapper.target:
                obj.confidence = (0.9 * 2 + obj.confidence) / 3
            else:
                obj.confidence = 0.9 / (0.9 + obj.confidence)
            if res == "unknown":
                obj.confidence = 0.0

            if not isinstance(obj.confidence, torch.Tensor):
                obj.confidence = torch.tensor(obj.confidence)
            obj.rgb = final_img
            obj.conf_list[obj.tag] = obj.confidence

            camera_points_real = project_to_camera(obj.pcd, self.mapper.camera_intrinsic, final_position,
                                              final_rotation)
            camera_points_real = np.array(camera_points_real)
            camera_points_real = camera_points_real.T
            camera_points_real = np.array(camera_points_real[:, :2], dtype=np.int32)
            flag = (camera_points_real[:, 0] >= 0) & (camera_points_real[:, 0] < 640) & \
                     (camera_points_real[:, 1] >= 0) & (camera_points_real[:, 1] < 480)
            camera_points_real = camera_points_real[flag]

            if camera_points_real.shape[0] != 0:
                bbox_real = np.array([np.min(camera_points_real, axis=0), np.max(camera_points_real, axis=0)])
                bbox_real = np.array([bbox_real[0][0], bbox_real[0][1], bbox_real[1][0], bbox_real[1][1]])
                obj.bbox = bbox_real

        # need_to_del = []
        # for (i, obj) in enumerate(real_C_objs):
        #     points = obj.pcd.point.positions.cpu().numpy()
        #     dist = np.linalg.norm(points - self.mapper.current_position, axis=1)
        #     flag = (dist <= 6.5) & (dist > 0.55)
        #     if np.sum(flag) <= 10:
        #         need_to_del.append(i)
        #         continue
        #     pcd_selected = np.where(flag)[0]
        #     pcd_selected = o3d.core.Tensor(pcd_selected, dtype=o3d.core.Dtype.Int64, device=self.mapper.pcd_device)
        #     obj.pcd = obj.pcd.select_by_index(pcd_selected)
        #     obj.position = np.mean(obj.pcd.point.positions.cpu().numpy(), axis=0)
        #     obj.num_list[obj.tag] = obj.pcd.point.positions.cpu().numpy().shape[0]
        #
        # real_C_objs = [obj for idx, obj in enumerate(real_C_objs) if idx not in need_to_del]

        # move all self.mapper.target_objects to the end
        real_C_objs_sorted = []
        target_objs = []
        for obj in real_C_objs:
            if obj.tag == self.mapper.target:
                target_objs.append(obj)
            else:
                real_C_objs_sorted.append(obj)
        real_C_objs_sorted += target_objs
        real_C_objs = real_C_objs_sorted

        self.mapper.objects, obj_indices = self.mapper.associate_object_entities(
            self.mapper.objects, real_C_objs)
        self.mapper.current_obj_indices += obj_indices
        self.object_pcd = self.mapper.update_object_pcd()

        self.mapper.current_obj_indices = list(set(self.mapper.current_obj_indices))

    def concat_panoramic(self, images):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 != 0:
                row = (i // 6)
                col = ((i % 6) // 2)
                copy_images[i] = cv2.putText(copy_images[i], "Direction %d" % i, (100, 100),
                                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6,
                                             cv2.LINE_AA)
                background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height +
                                 height:, col * width + col * 10:col * width + col * 10 +
                                 width, :] = copy_images[i]

        return background_image

    def update_trajectory(self, on_node_flag=False):
        self.episode_steps += 1
        self.metrics = self.env.get_metrics()
        self.rgb_trajectory.append(cv2.cvtColor(self.obs['rgb'], cv2.COLOR_BGR2RGB))
        self.depth_trajectory.append((self.obs['depth'] / 5.0 * 255.0).astype(np.uint8))
        self.temporary_depths.append(self.obs['depth'].copy())

        topdown_image = cv2.cvtColor(
            colorize_draw_agent_and_fit_to_height(self.metrics['top_down_map'], 1024),
            cv2.COLOR_BGR2RGB)
        topdown_image = cv2.flip(topdown_image, 0)
        text = f"Success:{self.metrics['success']:.2f}, SPL:{self.metrics['spl']:.2f}, SoftSPL:{self.metrics['soft_spl']:.2f}, DTS:{self.metrics['distance_to_goal']:.2f}, Step:{self.episode_steps}, Goal:{self.env.current_episode.object_category}"
        topdown_image = cv2.putText(
            topdown_image, text, (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        self.topdown_trajectory.append(topdown_image)

        self.position_trajectory.append(self.position)
        self.rotation_trajectory.append(self.rotation)

        if len(self.position_trajectory) > 1:
            pos1 = self.position_trajectory[-1]
            pos2 = self.position_trajectory[-2]
            pos1 = [pos1[0], pos1[2]]
            pos2 = [pos2[0], pos2[2]]
            self.travel_distance += np.linalg.norm((np.array(pos1) - np.array(pos2)))

        self.mapper.update(self.rgb_trajectory[-1], self.obs['depth'], self.position, self.rotation, self.episode_samples - 1, self.episode_steps, on_node_flag, self.current_node_idx)

        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/rgb', exist_ok=True)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/depth', exist_ok=True)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/topdown', exist_ok=True)
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/rgb/monitor-rgb_{self.episode_steps}.jpg",
            self.rgb_trajectory[-1])
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/depth/monitor-depth_{self.episode_steps}.jpg",
            self.depth_trajectory[-1])
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/topdown/monitor-topdown_{self.episode_steps}.jpg",
            self.topdown_trajectory[-1])
        # cv2.imwrite("monitor-rgb.jpg", self.rgb_trajectory[-1])

        if self.episode_steps == 499:
            self.obs = self.env.step(0)
            self.update_trajectory()
            logger.info('Episode over!!!!!')

    def save_trajectory(self, dir="./tmp_objnav/"):
        import imageio
        os.makedirs(dir, exist_ok=True)

        self.mapper.save_pointcloud_debug(dir)
        fps_writer = imageio.get_writer(dir + "fps.mp4", fps=4)
        dps_writer = imageio.get_writer(dir + "depth.mp4", fps=4)
        # seg_writer = imageio.get_writer(dir+"segmentation.mp4", fps=4)
        metric_writer = imageio.get_writer(dir + "metrics.mp4", fps=4)
        # for i,img,dep,seg,met in zip(np.arange(len(self.rgb_trajectory)),self.rgb_trajectory,self.depth_trajectory,self.segmentation_trajectory,self.topdown_trajectory):
        for i, img, dep, met in zip(np.arange(len(self.rgb_trajectory)), self.rgb_trajectory,
                                    self.depth_trajectory, self.topdown_trajectory):
            # for i, img, dep, met in zip(np.arange(len(self.rgb_trajectory)), self.rgb_trajectory,
            #                             self.depth_trajectory, self.topdown_trajectory):
            fps_writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            dps_writer.append_data(dep)
            # seg_writer.append_data(cv2.cvtColor(seg,cv2.COLOR_BGR2RGB))
            metric_writer.append_data(cv2.cvtColor(met, cv2.COLOR_BGR2RGB))

        # for index,pano_img in enumerate(self.panoramic_trajectory):
        #     cv2.imwrite(dir+"%d-pano.jpg"%index,pano_img)
        # with open(dir+"gpt4_history.txt",'w') as file:
        #     file.write("".join(self.gpt_trajectory))
        # with open(dir+"gpt4v_history.txt",'w') as file:
        #     file.write("".join(self.gptv_trajectory))

        # for i,afford,safford,hafford,cafford,gafford,oafford in zip(np.arange(len(self.affordance_trajectory)),self.affordance_trajectory,self.semantic_affordance_trajectory,self.history_affordance_trajectory,self.action_affordance_trajectory,self.gpt4v_affordance_trajectory,self.obstacle_affordance_trajectory):
        #     o3d.io.write_point_cloud(dir+"afford-%d-plan.ply"%i,afford)
        #     o3d.io.write_point_cloud(dir+"semantic-afford-%d-plan.ply"%i,safford)
        #     o3d.io.write_point_cloud(dir+"history-afford-%d-plan.ply"%i,hafford)
        #     o3d.io.write_point_cloud(dir+"action-afford-%d-plan.ply"%i,cafford)
        #     o3d.io.write_point_cloud(dir+"gpt4v-afford-%d-plan.ply"%i,gafford)
        #     o3d.io.write_point_cloud(dir+"obstacle-afford-%d-plan.ply"%i,oafford)

        fps_writer.close()
        dps_writer.close()
        # seg_writer.close()
        metric_writer.close()

        # rm the top-down folder
        folder_path = f'{self.save_dir}/episode-{self.episode_samples-1}/topdown'
        os.system(f'rm -r {folder_path}')

    def make_plan(self, rotate=True, failed=False):
        if rotate == True:
            self.rotate_panoramic()
        self.chainon_answer = self.query_chainon()
        self.gpt4v_answer = self.query_gpt4v()
        self.gpt4v_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
        self.gpt4v_pcd = gpu_merge_pointcloud(self.gpt4v_pcd, self.temporary_pcd[self.gpt4v_answer])
        self.found_goal = bool(self.chainon_answer['Flag'])
        self.affordance_pcd, self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(
            self.chainon_answer['Action'],
            self.chainon_answer['Landmark'],
            self.gpt4v_pcd,
            self.chainon_answer['Flag'],
            failure_mode=failed)
        self.semantic_afford, self.history_afford, self.action_afford, self.gpt4v_afford, self.obs_afford = self.mapper.get_debug_affordance_map(
            self.chainon_answer['Action'], self.chainon_answer['Landmark'], self.gpt4v_pcd)
        if self.affordance_pcd.max() == 0:
            self.affordance_pcd, self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(
                self.chainon_answer['Action'],
                self.chainon_answer['Landmark'],
                self.gpt4v_pcd,
                False,
                failure_mode=failed)
            self.found_goal = False

        self.affordance_map, self.colored_affordance_map = project_costmap(
            self.mapper.navigable_pcd, self.affordance_pcd, self.mapper.grid_resolution)
        self.target_point = self.mapper.navigable_pcd.point.positions[
            self.affordance_pcd.argmax()].cpu().numpy()
        self.plan_position = self.mapper.current_position.copy()
        target_index = translate_point_to_grid(self.mapper.navigable_pcd, self.target_point,
                                               self.mapper.grid_resolution)
        start_index = translate_point_to_grid(self.mapper.navigable_pcd,
                                              self.mapper.current_position,
                                              self.mapper.grid_resolution)
        self.path = path_planning(self.affordance_map, start_index, target_index)
        self.path = [
            translate_grid_to_point(self.mapper.navigable_pcd,
                                    np.array([[waypoint.y, waypoint.x, 0]]),
                                    self.mapper.grid_resolution)[0] for waypoint in self.path
        ]
        if len(self.path) == 0:
            self.waypoint = self.mapper.navigable_pcd.point.positions.cpu().numpy()[np.argmax(
                self.affordance_pcd)]
            self.waypoint[2] = self.mapper.current_position[2]
        elif len(self.path) < 5:
            self.waypoint = self.path[-1]
            self.waypoint[2] = self.mapper.current_position[2]
        else:
            self.waypoint = self.path[4]
            self.waypoint[2] = self.mapper.current_position[2]

        self.affordance_trajectory.append(self.colored_affordance_pcd)
        self.obstacle_affordance_trajectory.append(self.obs_afford)
        self.semantic_affordance_trajectory.append(self.semantic_afford)
        self.history_affordance_trajectory.append(self.history_afford)
        self.action_affordance_trajectory.append(self.action_afford)
        self.gpt4v_affordance_trajectory.append(self.gpt4v_afford)

    def step(self):
        to_target_distance = np.sqrt(np.sum(np.square(self.mapper.current_position -
                                                      self.waypoint)))
        if to_target_distance < 0.6 and len(self.path) > 0:
            self.path = self.path[min(5, len(self.path) - 1):]
            if len(self.path) < 3:
                self.waypoint = self.path[-1]
                self.waypoint[2] = self.mapper.current_position[2]
            else:
                self.waypoint = self.path[2]
                self.waypoint[2] = self.mapper.current_position[2]

        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])

        # act = self.planner.get_next_action(pid_waypoint)

        move_distance = np.sqrt(np.sum(np.square(self.mapper.current_position -
                                                 self.plan_position)))
        if (act == 0 or move_distance > 3.0) and not self.found_goal:
            self.make_plan(rotate=True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
        if act == 0 and not self.found_goal:
            self.make_plan(False, True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
            logger.info("Warning: Failure locomotion and action = %d" % act)
        if not self.env.episode_over:
            self.obs = self.env.step(act)
            self.update_trajectory()

    def _merge_temporary_pointclouds(self):
        merged_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
        for pcd in self.temporary_pcd:
            merged_pcd = gpu_merge_pointcloud(merged_pcd, pcd)
        return merged_pcd

    def _save_obs_pointcloud(self, pcd, idx, step, path_idx=None):
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(pcd.point.positions.cpu().numpy())
        save_pcd.colors = o3d.utility.Vector3dVector(pcd.point.colors.cpu().numpy())
        os.makedirs(f'{self.save_dir}/episode-{idx}/obs', exist_ok=True)
        if path_idx is None:
            file_path = f'{self.save_dir}/episode-{idx}/obs/obs_{step}.ply'
        else:
            file_path = f'{self.save_dir}/episode-{idx}/obs/obs_{step}_{path_idx}.ply'
        o3d.io.write_point_cloud(file_path, save_pcd)

    def _log_mapper_state_before_after_get_nodes(self, step, node, idx):
        logger.info("\n \n --------------------------------------------------")
        logger.info(f'Current step: {step}')
        logger.info("before {}", self.mapper.node_cnt)
        self.mapper.get_nodes(self.temporary_pcd, self.angles, node, episode_idx=idx, step=step)
        logger.info("after {}", self.mapper.node_cnt)
        logger.info(self.mapper.get_nodes_states())
        logger.info(self.mapper.get_nodes_positions())
        logger.info("current position: {}", self.mapper.current_position)

    def make_plan_mod_no_relocate_no_gpt(self,
                                  rotate=True,
                                  failed=False,
                                  initial=False,
                                  node=None,
                                  idx=None):
        self.on_node_flag = True

        self.rotate_panoramic()
        if self.env.episode_over:
            return False, False

        self.current_pcd = self._merge_temporary_pointclouds()
        step = self.episode_steps
        self._save_obs_pointcloud(self.current_pcd, idx=idx, step=step)

        self._log_mapper_state_before_after_get_nodes(step, node, idx)
        self.mapper.update_obj(self.current_node_idx, self.mapper.current_obj_indices)

        logger.info("-------------------Check Whether The Object is Found-------------------")
        # self.found_goal, self.object_final = self.mapper.object_found(self.instruct_goal, idx=idx, step=step)
        self.found_goal, self.object_final = self.mapper.object_found_no_gpt(self.instruct_goal,
                                                                             idx=idx,
                                                                             step=step)
        if self.found_goal:
            # self.path = np.array([self.object_final.position])
            self.found_goal_position = self.mapper.current_position
            self.find_final_waypoint()
            self.whether_to_check_again()
            self.path[:, 2] = self.mapper.current_position[2]
            self.waypoint_final = None
            self.found_goal = True

            return True, self.found_goal

        current_node = self.mapper.nodes[self.current_node_idx]
        room_node = self.mapper.room_nodes[current_node.room_idx]
        self.waypoint = self.mapper.explore_in_room(room_node)
        if self.waypoint is not None:
            self.current_node_idx = self.waypoint.idx
            self.path = np.array([self.waypoint.position])
            self.path_index = 0
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.waypoint = self.path[0]


            return True, self.found_goal

        if self.waypoint is None:
            logger.info("----------------Relocate After Fully Explored----------------")
            # ----------------relocate----------------
            # self.waypoint_final, self.object_final, found_goal = self.mapper.get_candidate_node(self.instruct_goal, idx=idx, step=step)
            room_state = [room_node.state for room_node in self.mapper.room_nodes]
            if 0 not in room_state:
                logger.info("Fully Explored!!!!!")
                return False, self.found_goal

            self.room_final = self.mapper.get_candidate_room_fully_explored_no_gpt(self.instruct_goal,
                                                                            idx=idx,
                                                                            step=step)

            self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)
            if self.waypoint is None:
                logger.info("Fully Explored!!!!!")
                return False, self.found_goal

            self.current_node_idx = self.waypoint.idx
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')
            # if self.room_final is None:
            #     return False, self.found_goal

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.path, self.path_node_idx = self.mapper.get_path(self.waypoint)

            self.path[:, 2] = self.mapper.current_position[2]
            if len(self.path) == 1:
                self.waypoint = self.path[0]
                self.path_node_idx = self.path_node_idx[0]
                self.path_index = 0
            else:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
                self.path_index = 0

            logger.info(f'Path: {self.path}')

            return True, self.found_goal

    def make_plan_mod_no_relocate(self,
                                  rotate=True,
                                  failed=False,
                                  initial=False,
                                  node=None,
                                  idx=None):
        self.on_node_flag = True

        self.rotate_panoramic()
        if self.env.episode_over:
            return False, False

        self.current_pcd = self._merge_temporary_pointclouds()
        step = self.episode_steps
        self._save_obs_pointcloud(self.current_pcd, idx=idx, step=step)

        self._log_mapper_state_before_after_get_nodes(step, node, idx)
        self.mapper.update_obj(self.current_node_idx, self.mapper.current_obj_indices)

        logger.info("-------------------Check Whether The Object is Found-------------------")
        # self.found_goal, self.object_final = self.mapper.object_found(self.instruct_goal, idx=idx, step=step)
        self.found_goal, self.object_final = self.mapper.object_found_no_gpt(self.instruct_goal,
                                                                             idx=idx,
                                                                             step=step)
        if self.found_goal:
            # self.path = np.array([self.object_final.position])

            self.found_goal_position = self.mapper.current_position
            self.find_final_waypoint()
            self.whether_to_check_again()
            self.path[:, 2] = self.mapper.current_position[2]
            self.waypoint_final = None
            self.found_goal = True

            return True, self.found_goal

        current_node = self.mapper.nodes[self.current_node_idx]
        room_node = self.mapper.room_nodes[current_node.room_idx]
        self.waypoint = self.mapper.explore_in_room(room_node)
        if self.waypoint is not None:
            self.current_node_idx = self.waypoint.idx
            self.path = np.array([self.waypoint.position])
            self.path_index = 0
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.waypoint = self.path[0]


            return True, self.found_goal

        if self.waypoint is None:
            logger.info("----------------Relocate After Fully Explored----------------")
            # ----------------relocate----------------
            # self.waypoint_final, self.object_final, found_goal = self.mapper.get_candidate_node(self.instruct_goal, idx=idx, step=step)
            room_state = [room_node.state for room_node in self.mapper.room_nodes]
            if 0 not in room_state:
                logger.info("Fully Explored, Visit unvisited nodes!")
                self.waypoint = self.mapper.explore_after_fully_explored()
            else:
                self.room_final = self.mapper.get_candidate_room_fully_explored(self.instruct_goal,
                                                                                idx=idx,
                                                                                step=step)

                self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)

            if self.waypoint is None:
                logger.info("No unvisited nodes, Fully Explored!!!!!")
                return False, self.found_goal

            self.current_node_idx = self.waypoint.idx
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')
            # if self.room_final is None:
            #     return False, self.found_goal

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.path, self.path_node_idx = self.mapper.get_path(self.waypoint)

            self.path[:, 2] = self.mapper.current_position[2]
            if len(self.path) == 1:
                self.waypoint = self.path[0]
                self.path_node_idx = self.path_node_idx[0]
                self.path_index = 0
            else:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
                self.path_index = 0

            logger.info(f'Path: {self.path}')

            return True, self.found_goal

    def make_plan_mod_relocate(self,
                                  rotate=True,
                                  failed=False,
                                  initial=False,
                                  node=None,
                                  idx=None):
        self.on_node_flag = True

        self.rotate_panoramic()
        if self.env.episode_over:
            return False, False

        self.current_pcd = self._merge_temporary_pointclouds()
        step = self.episode_steps
        self._save_obs_pointcloud(self.current_pcd, idx=idx, step=step)

        self._log_mapper_state_before_after_get_nodes(step, node, idx)
        self.mapper.update_obj(self.current_node_idx, self.mapper.current_obj_indices)

        logger.info("-------------------Check Whether The Object is Found-------------------")
        # self.found_goal, self.object_final = self.mapper.object_found(self.instruct_goal, idx=idx, step=step)
        self.found_goal, self.object_final = self.mapper.object_found_no_gpt(self.instruct_goal,
                                                                             idx=idx,
                                                                             step=step)
        if self.found_goal:
            self.found_goal_position = self.mapper.current_position
            self.find_final_waypoint()
            self.whether_to_check_again()

            # self.waypoint = self.object_final.find_closest(self.found_goal_position)
            # self.waypoint[2] = self.found_goal_position[2]
            # self.path = np.array([self.waypoint])
            # self.path_index = 0

            self.path[:, 2] = self.mapper.current_position[2]
            self.waypoint_final = None
            self.found_goal = True

            return True, self.found_goal

        current_node = self.mapper.nodes[self.current_node_idx]
        room_node = self.mapper.room_nodes[current_node.room_idx]

        # # if enter a new room, decide whether to go back
        if self.waypoint is not None:
            previous_node = self.mapper.nodes[self.current_node_idx]
            if previous_node.room_idx != current_node.room_idx:
                logger.info('Accidentally enter a new room!!!')

        self.waypoint = self.mapper.explore_in_room_relocate(room_node)
        if self.waypoint is not None:
            self.current_node_idx = self.waypoint.idx
            self.path = np.array([self.waypoint.position])
            self.path_index = 0
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.waypoint = self.path[0]

            return True, self.found_goal

        if self.waypoint is None:
            logger.info("----------------Relocate----------------")
            # ----------------relocate----------------
            # self.waypoint_final, self.object_final, found_goal = self.mapper.get_candidate_node(self.instruct_goal, idx=idx, step=step)
            room_state = [room_node.state for room_node in self.mapper.room_nodes]
            if 0 not in room_state:
                logger.info("Fully Explored, Visit unvisited nodes!")
                self.waypoint = self.mapper.explore_after_fully_explored()
            else:
                self.room_final = self.mapper.get_candidate_room_fully_explored(self.instruct_goal,
                                                                                idx=idx,
                                                                                step=step)

                self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)

            if self.waypoint is None:
                logger.info("No unvisited nodes, Fully Explored!!!!!")
                return False, self.found_goal

            self.current_node_idx = self.waypoint.idx
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')
            # if self.room_final is None:
            #     return False, self.found_goal

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.path, self.path_node_idx = self.mapper.get_path(self.waypoint)

            self.path[:, 2] = self.mapper.current_position[2]
            if len(self.path) == 1:
                self.waypoint = self.path[0]
                self.path_node_idx = self.path_node_idx[0]
                self.path_index = 0
            else:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
                self.path_index = 0

            logger.info(f'Path: {self.path}')

            return True, self.found_goal

    def make_plan_mod_process(self,
                              rotate=True,
                              failed=False,
                              initial=False,
                              node=None,
                              idx=None,
                              path_idx=None):
        self.on_node_flag = True

        if self.mapper.process_obs_pcd.is_empty():
            return

        step = self.episode_steps
        self._save_obs_pointcloud(self.mapper.process_obs_pcd, idx=idx, step=step, path_idx=path_idx)

        self.mapper.get_nodes_process(node, idx=idx, step=step, path_idx=path_idx)

    def check_again(self, episode_step):
        current_depth = self.obs['depth'].copy()
        camera_points = project_to_camera(self.object_final.pcd, self.mapper.camera_intrinsic,
                                          self.mapper.current_position,
                                          self.mapper.current_rotation)
        camera_points = np.array(camera_points)
        camera_points = camera_points.T
        depth = np.array(camera_points[:, 2], dtype=np.float32)
        camera_points = np.array(camera_points[:, :2], dtype=np.int32)
        flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
               (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
        camera_points = camera_points[flag]
        depth = depth[flag]

        current_depth = current_depth[camera_points[:, 1], camera_points[:, 0]][:, 0]
        current_depth = np.array(current_depth, dtype=np.float32)
        depth_flag = (depth - current_depth) < 0.2
        camera_points = camera_points[depth_flag]

        if len(camera_points) == 0:
            logger.info(f"Abort check again due to visibility.")
            return True

        bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
        # x1 y1 x2 y2
        bbox = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
        bbox = torch.tensor(bbox).unsqueeze(0)

        img = self.rgb_trajectory[-1].copy()

        img_vis = visualize_mask(img, bbox)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/check_again', exist_ok=True)
        cv2.imwrite(
            f'{self.save_dir}/episode-{self.episode_samples-1}/check_again/rgb_{episode_step}.jpg',
            img_vis)
        return check_again_object_in_bbox(
            img_vis=img_vis,
            target=self.mapper.target,
            save_dir=self.save_dir,
            episode_idx=self.episode_samples - 1,
            episode_step=episode_step,
            vlm=self.vlm,
        )

    def step_mod(self, idx):
        if self.episode_steps == 499:
            self.obs = self.env.step(0)
            self.update_trajectory()
            logger.info('Episode over!!!!!')
            return False

        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])

        current_position = self.mapper.current_position + self.mapper.initial_position
        current_position = np.array([
            current_position[0], self.mapper.initial_position[2] - 0.88, current_position[1]
        ])
        geo_distance = self.env.sim.geodesic_distance(current_position, pid_waypoint)
        logger.info(f'Geo distance: {geo_distance}')

        # tmp = habitat_translation(self.obs['gps'])-self.mapper.initial_position
        # print(self.obs['gps'])
        # print(self.mapper.initial_position)
        # print(self.mapper.current_position)
        if self.found_goal:
            logger.info("Found goal!!!")
            logger.info("Current position: {}", self.mapper.current_position)
            logger.info("Goal position: {}", self.object_final.position)

            positions = self.object_final.pcd.point.positions.cpu().numpy()
            to_target_distance = np.min(
                np.linalg.norm(positions[:, :2] - self.mapper.current_position[:2], axis=1))
            logger.info("Distance to goal: {}", to_target_distance)

            if self.need_check_again:
                to_check_again_distance = self.calculate_geo_distance(self.check_again_postion, self.mapper.current_position)
                if to_check_again_distance > 0.5:
                    act = self.planner.get_next_action(pid_waypoint)
                else:
                    # check again
                    self.need_check_again = False

                    final_pos = self.object_final.position[:2]
                    nw_pos = self.mapper.current_position[:2]
                    nw_ori = self.rotation
                    nw_ori = habitat_rotation(nw_ori)
                    nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
                    final_pos = final_pos - nw_pos

                    nw_ori = nw_ori / np.linalg.norm(nw_ori)
                    final_pos = final_pos / np.linalg.norm(final_pos)
                    dot_product = np.clip(np.dot(nw_ori, final_pos), -1.0, 1.0)
                    angle_rad = np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)

                    cross_product = np.cross(nw_ori, final_pos)
                    direction = 3 if cross_product > 0 else 2
                    num_turns = int(np.round(angle_deg / 30))

                    logger.info("Now step: {}", self.episode_steps)
                    logger.info("Direction: {}", direction)
                    logger.info("Num turns: {}", num_turns)

                    for i in range(num_turns):
                        self.obs = self.env.step(direction)
                        self.update_trajectory(self.on_node_flag)
                        if self.env.episode_over:
                            return False

                    self.found_goal = self.check_again(self.episode_steps)
                    logger.info("Check again at step: {}", self.episode_steps)
                    logger.info("Check again: {}", self.found_goal)

                    if not self.found_goal:
                        self.after_check_again()
                    else:
                        act = self.planner.get_next_action(pid_waypoint)
            else:
                if to_target_distance > self.success_distance * self.stop_criterion:
                    act = self.planner.get_next_action(pid_waypoint)
                else:
                    act = 0

        # not use else because self.check_again may change the self.found_goal
        if not self.found_goal:
            act = self.planner.get_next_action(pid_waypoint)
            while act == 0 and not self.found_goal:
                self.path_index += 1
                if self.path_index >= len(self.path):
                    # get to the next waypoint we want
                    if self.relocate:
                        flag, self.found_goal = self.make_plan_mod_relocate(
                            rotate=True, idx=idx, node=self.waypoint_final)
                    else:
                        if self.gpt_relocate:
                            flag, self.found_goal = self.make_plan_mod_no_relocate(
                                rotate=True, idx=idx, node=self.waypoint_final)
                        else:
                            flag, self.found_goal = self.make_plan_mod_no_relocate_no_gpt(
                                rotate=True, idx=idx, node=self.waypoint_final)
                    if self.env.episode_over:
                        return False

                    if not flag and not self.env.episode_over:
                        self.obs = self.env.step(act)
                        self.update_trajectory(self.on_node_flag)
                        logger.info(self.env.episode_over)
                        return False
                else:
                    logger.info(f'!!!!!!!!!!!!!!!!Bug: enter make_plan_mod_process')
                    self.obs = self.env.step(0)
                    self.update_trajectory(self.on_node_flag)
                    logger.info(self.env.episode_over)
                    return False
                    # self.make_plan_mod_process(idx=idx, node=None, path_idx=self.path_index)

                self.waypoint = self.path[self.path_index]
                self.waypoint[2] = self.mapper.current_position[2]

                # self.waypoint = [-6.1704755, -1.333439, -0.8]

                logger.info(f'Waypoint: {self.waypoint}')
                pid_waypoint = self.waypoint + self.mapper.initial_position
                pid_waypoint = np.array(
                    [pid_waypoint[0],
                     self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                act = self.planner.get_next_action(pid_waypoint)

            if act == 1:
                self.on_node_flag = False

        logger.info("Step: {}", self.episode_steps)
        logger.info("Next Action: {}", act)
        logger.info("Episode over: {}", self.env.episode_over)
        logger.info("Found goal: {}", self.found_goal)
        logger.info("Waypoint: {}", self.waypoint)

        if not self.env.episode_over:
            if self.found_goal and act == 0:
                final_check_flag = self.final_check()
                if not final_check_flag:
                    pid_waypoint = self.waypoint + self.mapper.initial_position
                    pid_waypoint = np.array(
                        [pid_waypoint[0],
                         self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                    act = self.planner.get_next_action(pid_waypoint)

            logger.info("Next Action: {}", act)
            self.obs = self.env.step(act)
            self.update_trajectory(self.on_node_flag)

            logger.info(self.env.episode_over)

            return True

    def to_json(self):
        return self.mapper.to_json()

    def whether_to_check_again(self):

        positions = self.object_final.pcd.point.positions.cpu().numpy()

        # scale = 0.
        # for i in range(3):
        #     scale = max(scale, np.max(positions[:, i]) - np.min(positions[:, i]))
        # logger.info("Scale: {}", scale)
        # self.best_distance = max(1.5 * scale, self.success_distance + 0.05)
        bbox = self.object_final.bbox
        obj_bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        obj_bbox_length = (bbox[2] - bbox[0])
        obj_bbox_width = (bbox[3] - bbox[1])

        logger.info(f'Need to check again')
        self.need_check_again = True
        # find the best point to check
        pathfinder = self.env.sim.pathfinder
        import habitat_sim
        path_request = habitat_sim.ShortestPath()
        current_position = self.found_goal_position + self.mapper.initial_position
        current_position = np.array([
            current_position[0], self.env.sim.get_agent_state().position[1], current_position[1]
        ])
        path_request.requested_start = current_position
        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
        path_request.requested_end = pid_waypoint

        # 计算最短路径
        found_path = pathfinder.find_path(path_request)
        points = path_request.points
        logger.info(f'Path: {points}')
        points = np.array(points)
        # swithc the y and z axis
        points = np.array([points[:, 0], points[:, 2], points[:, 1]]).T
        points = points - self.mapper.initial_position
        points[:, 2] = self.found_goal_position[2] + 0.88
        logger.info(f"Path points: {points}")

        interpolated_path = []
        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]
            # use np.linspace to interpolate the path between the two points
            distance = np.linalg.norm(point1 - point2)
            num_points = max(1, int(distance / 0.25))
            interpolated_points = [(1 - t) * point1 + t * point2 for t in np.linspace(0, 1, num_points)]
            interpolated_path.extend(interpolated_points)
        stop_idx = None
        for point_idx, point in enumerate(interpolated_path):
            to_target_distance = np.min(np.linalg.norm(positions[:, :2] - point[:2], axis=1))
            if to_target_distance <= self.success_distance * self.stop_criterion:
                stop_idx = point_idx
        if stop_idx is None:
            stop_idx = len(interpolated_path) - 1
        interpolated_path = interpolated_path[:stop_idx + 1]

        logger.info(f"Interpolated_path: {interpolated_path}")

        # invert the path
        interpolated_path = interpolated_path[::-1]
        final_pos = self.object_final.position[:2]
        for point_idx, point in enumerate(interpolated_path):
            position = point
            stop_pos = position[:2]
            orient = final_pos - stop_pos
            orient = orient / np.linalg.norm(orient)
            rotation_matrix = np.array([[-orient[1], 0, -orient[0]],
                                        [orient[0], 0, -orient[1]],
                                        [0, 1, 0]])
            camera_points = project_to_camera(self.object_final.pcd, self.mapper.camera_intrinsic,
                                              position,
                                              rotation_matrix)
            camera_points = np.array(camera_points)
            all_pc_number = camera_points.shape[1]
            camera_points = camera_points.T
            depth = np.array(camera_points[:, 2], dtype=np.float32)
            camera_points = np.array(camera_points[:, :2], dtype=np.int32)
            flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
                   (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
            camera_points = camera_points[flag]
            depth = depth[flag]

            if len(camera_points) == 0:
                continue

            bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
            bbox_area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
            length = (bbox[1][0] - bbox[0][0])
            width = (bbox[1][1] - bbox[0][1])

            # # x1 y1 x2 y2
            # bbox1 = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
            # bbox1 = torch.tensor(bbox1).unsqueeze(0)
            # depth_vis = np.zeros((480, 640), dtype=np.float32)
            # depth_vis[camera_points[:, 1], camera_points[:, 0]] = depth
            # depth_vis = depth_vis * 255
            # # expand the depth image to 3 channels
            # depth_vis = np.stack([depth_vis] * 3, axis=-1)
            # img = depth_vis
            #
            # img_vis = visualize_mask(img, bbox1)
            # os.makedirs(f'{self.save_dir}/episode-{self.episode_samples - 1}/check_again_debug', exist_ok=True)
            # cv2.imwrite(
            #     f'{self.save_dir}/episode-{self.episode_samples - 1}/check_again_debug/rgb_{point_idx}.jpg',
            #     img_vis)

            visible_pc_number = len(camera_points)
            logger.info(f'At {position}, Visible Ratio: {visible_pc_number / all_pc_number}, Box Ratio: {bbox_area / obj_bbox_area}')
            if visible_pc_number > 0.95 * all_pc_number and bbox_area > obj_bbox_area:
                big_ratio = np.sqrt(bbox_area / obj_bbox_area)
                if length > 0.8 * big_ratio * obj_bbox_length and width > 0.8 * big_ratio * obj_bbox_width:
                    # this is the best point to check
                    logger.info(f'Found a good point to check: {position}, Visible Ratio: {visible_pc_number / all_pc_number}, Box Ratio: {bbox_area / obj_bbox_area}')
                    self.check_again_postion = position
                    return

        # if we can't find a good point to check, then use the last point
        if len(interpolated_path) > 0:
            logger.info(f"Can't find a good point to check, use the last point")
            self.check_again_postion = interpolated_path[-1]
        else:
            logger.info(f"Can't find a good point to check, use the current position")
            self.check_again_postion = self.mapper.current_position



            # current_depth = self.obs['depth'].copy()
            # camera_points = project_to_camera(self.object_final.pcd, self.mapper.camera_intrinsic,
            #                                   self.mapper.current_position,
            #                                   self.mapper.current_rotation)
            # camera_points = np.array(camera_points)
            # camera_points = camera_points.T
            # depth = np.array(camera_points[:, 2], dtype=np.float32)
            # camera_points = np.array(camera_points[:, :2], dtype=np.int32)
            # flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
            #        (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
            # camera_points = camera_points[flag]
            # depth = depth[flag]
            #
            # current_depth = current_depth[camera_points[:, 1], camera_points[:, 0]][:, 0]
            # current_depth = np.array(current_depth, dtype=np.float32)
            # depth_flag = np.abs(current_depth - depth) < 0.1
            # camera_points = camera_points[depth_flag]
            #
            # if len(camera_points) == 0:
            #     return False
            #
            # bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
            #
            # # img = self.rgb_trajectory[-1].copy()
            # # cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 255, 0), 2)
            # # os.makedirs(f'{self.save_dir}/episode-{self.episode_samples - 1}/check', exist_ok=True)
            # # cv2.imwrite(
            # #     f"{self.save_dir}/episode-{self.episode_samples - 1}/check/check_{self.episode_steps}.jpg",
            # #     img)
            #
            # bbox_area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
            # logger.info(f'{bbox_area}, {self.bbox_area}')
            # if bbox_area > 1.2 * self.bbox_area:
            #     return True
            # else:
            #     return False

    def after_check_again(self):
        self.object_final.tag = "nothing"
        self.object_final.confidence = torch.tensor(1.0)
        self.object_final.conf_list = {"nothing": self.object_final.confidence}
        self.object_final.num_list = {"nothing": 10000}

        waypoint_node = self.mapper.explore_after_check()
        if waypoint_node is None:
            logger.info("Fully Explored!!!!!")
            return 0

        self.waypoint_final = waypoint_node
        self.waypoint = waypoint_node.position
        self.waypoint[2] = self.mapper.current_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0

        if waypoint_node.state == 1:
            self.just_come_back = True
        else:
            self.just_come_back = False

    def final_check(self):
        final_point = self.mapper.current_position
        final_point[2] = final_point[2] + 0.88
        final_point_voxel_idx = translate_single_point_to_grid(final_point, self.mapper.grid_resolution,
                                                               self.mapper.voxel_dimension)

        voxels = np.zeros(self.mapper.voxel_dimension, dtype=np.int32)
        voxel_idxs = translate_point_to_grid(self.mapper.useful_pcd.point.positions.cpu().numpy(),
                                             self.mapper.grid_resolution, self.mapper.voxel_dimension)
        voxels[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1

        obj_voxel_idxs = translate_point_to_grid(self.object_final.pcd.point.positions.cpu().numpy(),
                                                 self.mapper.grid_resolution, self.mapper.voxel_dimension)
        # voxels[obj_voxel_idxs[:, 0], obj_voxel_idxs[:, 1], obj_voxel_idxs[:, 2]] = 0

        big_visible_flag = False
        for obj_voxel_idx in obj_voxel_idxs:
            ray_idxs = bresenham_3d(obj_voxel_idx, final_point_voxel_idx)[1:]
            small_visible_flag = True
            for ray_idx in ray_idxs:
                if ray_idx[0] < 0 or ray_idx[0] >= self.mapper.voxel_dimension[0] or \
                        ray_idx[1] < 0 or ray_idx[1] >= self.mapper.voxel_dimension[1] or \
                        ray_idx[2] < 0 or ray_idx[2] >= self.mapper.voxel_dimension[2]:
                    continue
                if voxels[ray_idx[0], ray_idx[1], ray_idx[2]] == 1:
                    small_visible_flag = False
                    continue

            if small_visible_flag:
                big_visible_flag = True
                break

        if big_visible_flag:
            # if the object is visible, then stop
            logger.info(f"Object is visible, Robot can stop at {final_point}")
            return True
        else:
            logger.info(f"!!!!!!!!!!!Can't see the object, because of occlusion.")
            pid_waypoint = self.found_goal_position + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            logger.info(f'First go back to the waypoint finding the object.')
            while True:
                act = self.planner.get_next_action(pid_waypoint)
                if act == 0:
                    break
                logger.info("Step: {}", self.episode_steps)
                logger.info("Next Action: {}", act)
                logger.info("Episode over: {}", self.env.episode_over)
                logger.info("Found goal: {}", self.found_goal)
                logger.info("Waypoint: {}", self.found_goal_position)

                self.obs = self.env.step(act)
                self.update_trajectory(self.on_node_flag)

            self.find_final_waypoint()

            return False


    def find_final_waypoint(self):
        waypoint_tmp = self.object_final.find_closest(self.found_goal_position)
        self.path = np.array([waypoint_tmp])
        non_valid_stop = True

        iter_num = 0
        distance_to_initial_waypoint = np.linalg.norm(waypoint_tmp[:2] - self.found_goal_position[:2])
        max_iter_num = min(8, int(distance_to_initial_waypoint / 0.1))
        while non_valid_stop and iter_num < max_iter_num:
            non_valid_stop = False
            pathfinder = self.env.sim.pathfinder

            # 设置一个 ShortestPath 查询
            import habitat_sim
            path_request = habitat_sim.ShortestPath()
            current_position = self.found_goal_position + self.mapper.initial_position
            current_position = np.array([
                current_position[0], self.mapper.initial_position[2] - 0.88, current_position[1]
            ])
            path_request.requested_start = current_position
            pid_waypoint = waypoint_tmp + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            path_request.requested_end = pid_waypoint

            # 计算最短路径
            found_path = pathfinder.find_path(path_request)
            if found_path:
                points = path_request.points
                # logger.info(f'Path: {points}')
                points = np.array(points)
                # swithc the y and z axis
                points = np.array([points[:, 0], points[:, 2], points[:, 1]]).T
                points = points - self.mapper.initial_position
                points[:, 2] = -0.8
                logger.info(f"Path points: {points}")

                interpolated_path = []
                for i in range(len(points) - 1):
                    point1 = points[i]
                    point2 = points[i + 1]
                    # use np.linspace to interpolate the path between the two points
                    distance = np.linalg.norm(point1 - point2)
                    num_points = max(1, int(distance / 0.25))
                    interpolated_points = [(1 - t) * point1 + t * point2 for t in np.linspace(0, 1, num_points)]
                    interpolated_path.extend(interpolated_points)
                # invert the path
                interpolated_path = interpolated_path[::-1]
                # find the first point in interpolated_path that is far away from the self.obj_final from a certain distance
                positions = self.object_final.pcd.point.positions.cpu().numpy()
                final_point = None
                for point in interpolated_path:
                    to_target_distance = np.min(np.linalg.norm(positions[:, :2] - point[:2], axis=1))
                    if to_target_distance > self.success_distance * self.stop_criterion:
                        final_point = point
                        break
                if final_point is None:
                    final_point = interpolated_path[-1]

                # # save final_point as a ply file
                # save_pcd = o3d.geometry.PointCloud()
                # save_pcd.points = o3d.utility.Vector3dVector(final_point.reshape(-1,3))
                # save_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
                # os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/depth_final', exist_ok=True)
                # o3d.io.write_point_cloud(f'{self.save_dir}/episode-{self.episode_samples-1}/depth_final/final_point_{self.episode_steps}.ply', save_pcd)

                # final_poit is where the robot will stop, get the orientation at this point with robot facing the object
                final_pos = self.object_final.position[:2]
                stop_pos = final_point[:2]
                final_pos = final_pos - stop_pos
                final_pos = final_pos / np.linalg.norm(final_pos)
                rotation_matrix = np.array([[-final_pos[1], 0, -final_pos[0]],
                                             [final_pos[0], 0, -final_pos[1]],
                                              [0, 1, 0]])

                final_point[2] = self.found_goal_position[2] + 0.88
                final_point_voxel_idx = translate_single_point_to_grid(final_point, self.mapper.grid_resolution, self.mapper.voxel_dimension)

                voxels = np.zeros(self.mapper.voxel_dimension, dtype=np.int32)
                voxel_idxs = translate_point_to_grid(self.mapper.useful_pcd.point.positions.cpu().numpy(), self.mapper.grid_resolution, self.mapper.voxel_dimension)
                voxels[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1

                obj_voxel_idxs = translate_point_to_grid(self.object_final.pcd.point.positions.cpu().numpy(), self.mapper.grid_resolution, self.mapper.voxel_dimension)
                # voxels[obj_voxel_idxs[:, 0], obj_voxel_idxs[:, 1], obj_voxel_idxs[:, 2]] = 0

                big_visible_flag = False
                for obj_voxel_idx in obj_voxel_idxs:
                    ray_idxs = bresenham_3d(obj_voxel_idx, final_point_voxel_idx)[1:]
                    small_visible_flag = True
                    for ray_idx in ray_idxs:
                        if ray_idx[0] < 0 or ray_idx[0] >= self.mapper.voxel_dimension[0] or \
                                ray_idx[1] < 0 or ray_idx[1] >= self.mapper.voxel_dimension[1] or \
                                ray_idx[2] < 0 or ray_idx[2] >= self.mapper.voxel_dimension[2]:
                            continue
                        if voxels[ray_idx[0], ray_idx[1], ray_idx[2]] == 1:
                            small_visible_flag = False
                            continue

                    if small_visible_flag:
                        big_visible_flag = True
                        break

                if big_visible_flag:
                    # if the object is visible, then stop
                    non_valid_stop = False
                    logger.info(f"Object is visible, {waypoint_tmp} is a valid waypoint")
                    self.waypoint = waypoint_tmp
                    self.waypoint[2] = self.found_goal_position[2]
                    self.path = np.array([self.waypoint])
                    self.path_index = 0
                    return

                else:
                    non_valid_stop = True
                    logger.info(f"Object is not visible, {waypoint_tmp} is not a valid stop point")
                    # move the pid_waypoint closer to self.mapper.current_position
                    vector_tmp = (self.found_goal_position - waypoint_tmp) / np.linalg.norm(self.found_goal_position - waypoint_tmp)
                    waypoint_tmp = waypoint_tmp + vector_tmp * 0.1

            else:
                non_valid_stop = True
                logger.info(f"Path to {waypoint_tmp} is not found. {waypoint_tmp} is unreachable.")
                vector_tmp = (self.found_goal_position - waypoint_tmp) / np.linalg.norm(
                    self.found_goal_position - waypoint_tmp)
                waypoint_tmp = waypoint_tmp + vector_tmp * 0.1

            iter_num += 1

        logger.info(f"Strange case, can't find a valid waypoint.")
        logger.info(f"Use waypoint: {waypoint_tmp}")
        self.waypoint = waypoint_tmp
        self.waypoint[2] = self.found_goal_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0

        return

    def calculate_geo_distance(self, point1, point2):
        point1 = point1 + self.mapper.initial_position
        point1 = np.array(
            [point1[0],
             self.env.sim.get_agent_state().position[1], point1[1]])
        point2 = point2 + self.mapper.initial_position
        point2 = np.array([
            point2[0], self.env.sim.get_agent_state().position[1], point2[1]
        ])
        geodesic_distance = self.env.sim.geodesic_distance(point1, point2)

        return geodesic_distance