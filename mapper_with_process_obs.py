from loguru import logger
import numpy as np

from mapping_utils.geometry import *
from mapping_utils.preprocess import *
from mapping_utils.projection import *
from mapping_utils.transform import *
from mapping_utils.path_planning import *
from mapping_utils.numbers import *
from habitat_sim.utils.common import d3_40_colors_rgb
from constants import *
import open3d as o3d
from scipy.spatial import KDTree
from matplotlib.path import Path
import networkx as nx
import os
from mapping_utils.representation import *
import cv2
from cv_utils.image_perceiver import *
from llm_utils.nav_prompt_room import *

from pydantic import BaseModel
from openai import OpenAI


import math
import skimage
import habitat_sim
import torch
import heapq
from bresenham import bresenham


class Instruct_Mapper:

    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 voxel_dimension=[500, 500, 20],
                 floor_height=-0.8,
                 ceiling_height=0.8,
                 translation_func=habitat_translation,
                 rotation_func=habitat_rotation,
                 rotate_axis=[0, 1, 0],
                 perceiver="mmdinosam",
                 save_dir="tmp",
                 categories=[],
                 no_gpt_seg=False,
                 device='cuda:0',
                 env=None,
                 vlm='gemini'):
        self.device = device
        self.pcd_device = o3d.core.Device(device.upper())

        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.voxel_dimension = voxel_dimension
        self.grid_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        self.frontiers_considered = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))

        self.floor_height = floor_height
        self.ceiling_height = ceiling_height

        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.rotate_axis = np.array(rotate_axis)

        self.obj_number = 2
        self.target = ""
        self.target_list = []

        self.vlm = vlm
        self.frontier_thres = 6

        classes = []
        for obj in categories:
            classes.append(obj['name'])
        for obj in INTEREST_OBJECTS:
            if obj not in classes:
                classes.append(obj)
        logger.info("number of classes: {}".format(len(classes)))

        if perceiver == "glee":
            self.object_perceiver = GLEE_Perceiver(classes=classes, device=device)
        elif perceiver == "ramsam":
            self.object_perceiver = RAMDINOSAM_Perceiver(classes=classes, device=device)
        elif perceiver == "dinosam":
            self.object_perceiver = DINOSAM_Perceiver(classes=classes, device=device)
        elif perceiver == "mmdinosam":
            self.object_perceiver = MMDINOSAM_Perceiver(classes=classes,
                                                        no_gpt_seg=no_gpt_seg,
                                                        device=device)
        else:
            raise NotImplementedError("Image perceiver not implemented")

        self.save_dir = save_dir

        self.env = env

        self.lower_img, self.upper_img = calc_depth_img(camera_intrinsic, 0.51, 4.9)

    def initialize(self, position, rotation, env):
        self.env = env

        self.update_iterations = 0
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.rotation = rotation

        self.scene_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.traversable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.traversable_pcd_all = o3d.t.geometry.PointCloud(self.pcd_device)

        self.grid_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        self.frontiers_considered = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))

        self.current_global_frontier_map_idxs = []

        # graph representation
        self.nodes = []
        self.nodes_pos_to_idx = {}
        self.node_cnt = 0
        self.neighbors = []
        self.current_node_idx, _ = self.add_node(self.current_position, has_frontier=False)
        self.change_state(self.nodes[self.current_node_idx])

        self.objects = []
        self.exist_objects = []
        # each object: pointcloud, tag, confidence, description
        self.trajectory_nodes = []
        self.trajectory_position = []
        self.current_obj_indices = []

        self.traj = [0]

        # define the traversability map
        self.init_map()

        self.prev_position = None

        self.traj_on_trav = []

        self.target = ""
        self.target_list = []

        self.room_nodes = []



        h, w = 480, 640
        coord_map_left = np.zeros((h, w, 2), dtype=np.float32)
        coord_map_right = np.zeros((h, w, 2), dtype=np.float32)
        coord_map_middle = np.zeros((h, w, 2), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                coord_map_left[i, j, 0] = i
                coord_map_left[i, j, 1] = j
                coord_map_right[i, j, 0] = i
                coord_map_right[i, j, 1] = j
                coord_map_middle[i, j, 0] = i
                coord_map_middle[i, j, 1] = j

        def get_homo(theta):
            K = self.camera_intrinsic
            theta_rad = np.radians(theta)
            R = np.array([
                [np.cos(theta_rad), 0, np.sin(theta_rad)],
                [0, 1, 0],
                [-np.sin(theta_rad), 0, np.cos(theta_rad)]
            ])
            mat = K @ R @ np.linalg.inv(K)
            translate = np.array([
                [1, 0, w],
                [0, 1, 0],
                [0, 0, 1]
            ])
            mat = translate @ mat
            return mat

        H1 = get_homo(-30)
        H2 = get_homo(30)
        coord_map_left = cv2.warpPerspective(coord_map_left, H1, (w * 3, h))
        coord_map_right = cv2.warpPerspective(coord_map_right, H2, (w * 3, h))

        coord_map_left = coord_map_left[:, :w]
        coord_map_right = coord_map_right[:, w*2:]
        self.coord_maps = [coord_map_left, coord_map_middle, coord_map_right]

    def init_map(self):
        self.map_size = 960  # 960 * 0.05 = 48m
        self.map_resolution = 0.05  # 0.05m
        full_w, full_h = self.map_size, self.map_size
        self.trav_map = torch.zeros(full_w, full_h).float().to(self.device)
        self.collision_map = self.trav_map.cpu().numpy()
        self.goal_gps_map = self.trav_map.cpu().numpy()
        self.origins = np.zeros((2))

        self.collision_threshold = 0.08

        def init_map_and_pose():
            self.trav_map.fill_(0.)

        init_map_and_pose()

    def add_node(self, position, pcd=None, has_frontier=False, frontier_idxs=np.array([]), block_current=False):

        logger.info("---------- Checking ---------- {}", position)
        if len(self.nodes) > 1:
            current_node = self.nodes[self.current_node_idx]
            current_node_idx = current_node.idx
            # if only 1 node to be added, need to ensure it wont be blocked by the current node
            if block_current:
                nodes_positions = [node.position for node in self.nodes if node.idx != current_node_idx]
                nodes_idxs = [node.idx for node in self.nodes if node.idx != current_node_idx]
            else:
                nodes_positions = [node.position for node in self.nodes]
                nodes_idxs = [node.idx for node in self.nodes]
            nodes_positions = np.array(nodes_positions)
            nodes_positions = nodes_positions + self.initial_position
            nodes_positions[:, 2] = self.initial_position[2] - 0.88
            nodes_positions = nodes_positions[:, [0, 2, 1]]

            current_node_position = position + self.initial_position
            current_node_position = np.array([
                current_node_position[0], self.initial_position[2] - 0.88, current_node_position[1]
            ])
            distance = [
                self.env.sim.geodesic_distance(current_node_position, node_position)
                for node_position in nodes_positions
            ]
            if np.min(distance) < 1.35 * 0.7:
                block_node = self.nodes[nodes_idxs[np.argmin(distance)]]
                block_node_idx = block_node.idx
                logger.info(f'Node at {block_node.position} is blocking adding new node at {position}')
                # if not adding node, need to update the frontier to the node blocking it
                if has_frontier and block_node_idx!=current_node_idx and block_node.state==NodeState.UNEXPLORED:
                    logger.info(f'Updating frontier of node at {position} to node at {block_node.position}')
                    if block_node.has_frontier:
                        frontier_idxs = frontier_idxs.astype(int)
                        block_node.frontier_idxs = np.concatenate((block_node.frontier_idxs, frontier_idxs), axis=0)
                        block_node.has_frontier = True
                        block_node.state=NodeState.UNEXPLORED
                    else:
                        logger.info(f'Node at {block_node.position} has no frontier, move it to the new node at {position}')
                        frontier_idxs = frontier_idxs.astype(int)
                        block_node.frontier_idxs = np.concatenate((block_node.frontier_idxs, frontier_idxs), axis=0)
                        block_node.has_frontier = True
                        block_node.state=NodeState.UNEXPLORED
                        block_node.position = position
                        block_node.pcd = pcd


                    self.frontiers_considered[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1

                return self.node_cnt - 1, False

        logger.info("---------- adding node ---------- {}", position)
        position = position.copy()
        node = our_Node(None, None, pcd, position, None, self.node_cnt, has_frontier, frontier_idxs.astype(int))
        self.nodes.append(node)
        self.node_cnt += 1
        self.neighbors.append([])
        self.nodes_pos_to_idx[tuple(position)] = self.node_cnt - 1

        # update the grid_map
        if has_frontier:
            # print(frontier_idxs.shape) # 12,2
            frontier_idxs = frontier_idxs.astype(int)
            # concatenate self.frontiers_considered and frontier_idxs
            self.frontiers_considered[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1

        return self.node_cnt - 1, True

    def visit_node(self, node_idx):
        node_pos = self.nodes[node_idx].position
        logger.info(f"---------- visiting node ---------- {node_idx} at {node_pos}")
        for key, value in self.nodes_pos_to_idx.items():
            logger.info("{} {}", key, value)
        self.nodes[node_idx].state = NodeState.EXPLORED
        self.nodes[node_idx].has_frontier = False
        self.nodes[node_idx].has_true_frontier = False

    def update_node_frontier(self):
        for node in self.nodes:
            if node.has_frontier:
                frontier_idxs = node.frontier_idxs
                frontier_idxs = np.array(frontier_idxs).astype(int)
                frontier_idxs = [idx for idx in frontier_idxs if self.grid_map[idx[0], idx[1]] != 0]
                if len(frontier_idxs) == 0:
                    node.has_frontier = False
                    node.has_true_frontier = False
                    node.frontier_idxs = np.array(frontier_idxs).reshape((-1, 2))
                else:
                    node.has_frontier = True
                    node.frontier_idxs = np.array(frontier_idxs).reshape((-1, 2))
            else:
                node.has_frontier = False
                node.has_true_frontier = False
                node.frontier_idxs = np.array([]).reshape((-1, 2))

    def update_node_true_frontier(self):
        for node in self.nodes:
            if node.has_frontier:
                frontier_idxs = node.frontier_idxs
                frontier_idxs = np.array(frontier_idxs).astype(int)
                frontier_idxs = [idx for idx in frontier_idxs if self.grid_map[idx[0], idx[1]] == 2]
                if len(frontier_idxs) > 0:
                    node.has_true_frontier = True
                else:
                    node.has_true_frontier = False

    def update_room_state(self):
        for room_node in self.room_nodes:
            room_node.update_state()

    def add_edge(self, node1_idx, node2_idx):

        self.neighbors[node1_idx].append(node2_idx)
        self.neighbors[node2_idx].append(node1_idx)

    def remove_edge(self, node1_idx, node2_idx):

        self.neighbors[node1_idx].remove(node2_idx)
        self.neighbors[node2_idx].remove(node1_idx)

    def get_edges(self):
        edges = []
        for i in range(self.node_cnt):
            for j in self.neighbors[i]:
                if i < j:
                    edges.append((i, j))
        return edges

    def update_obj(self, current_node_idx, obj_indices):
        # each object connect all nodes in 2.5m
        dis_thres = 2.5

        node = self.nodes[current_node_idx]
        for ind in obj_indices:
            if ind in node.objects:
                continue

            nwdist = np.linalg.norm(self.objects[ind].position[:2] - node.position[:2])
            if nwdist < dis_thres:
                need_del = []
                for other_obj in self.objects[ind].nodes:
                    dis = np.linalg.norm(self.nodes[other_obj].position[:2] -
                                         self.objects[ind].position[:2])
                    if dis >= dis_thres:
                        need_del.append(other_obj)

                for other_obj in need_del:
                    self.nodes[other_obj].objects.remove(ind)
                    self.objects[ind].nodes.remove(other_obj)

                self.objects[ind].nodes.append(node.idx)
                node.objects.append(ind)
            else:
                if len(self.objects[ind].nodes) == 0:
                    self.objects[ind].nodes.append(node.idx)
                    node.objects.append(ind)

    def get_nodes_positions(self):
        return np.array([node.position for node in self.nodes])

    def get_nodes_states(self):
        return np.array([node.state for node in self.nodes])

    def find_closest_node(self, position):
        dist = []
        for node in self.nodes:
            dist.append(np.linalg.norm(node.position[:2] - position[:2]))
        dist = np.array(dist)
        return self.nodes[np.argmin(dist)]

    def find_closest_unexplored_node(self, node_pos):
        # find the closet node to any given node in the graph, distance is the sum of edge weights
        node = self.nodes[self.current_node_idx]
        node_indx = node.idx
        dist = np.full(self.node_cnt, np.inf)
        dist[node_indx] = 0

        pq = []
        heapq.heappush(pq, (0, node_indx))
        while pq:
            current_dist, u = heapq.heappop(pq)
            if self.nodes[u].state == NodeState.UNEXPLORED:
                return self.nodes[u]
            for v in self.neighbors[u]:
                alt = current_dist + np.linalg.norm(self.nodes[u].position[:2] -
                                                    self.nodes[v].position[:2])
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
        return None

    def find_the_closest_path(self, start, end):
        # Initialize distances and previous nodes
        start = self.find_closest_node(start)
        end = self.find_closest_node(end)

        start_idx = start.idx
        end_idx = end.idx

        dist = np.full(self.node_cnt, np.inf)
        prev = np.full(self.node_cnt, -1, dtype=int)
        dist[start_idx] = 0

        # Priority queue for Dijkstra's algorithm
        pq = []
        heapq.heappush(pq, (0, start_idx))  # (distance, node)

        while pq:
            current_dist, u_idx = heapq.heappop(pq)
            u = self.nodes[u_idx]
            # If we reach the end node, stop early
            if u == end:
                break

            # Skip if the distance is not optimal
            if current_dist > dist[u_idx]:
                continue

            # Iterate over neighbors
            for v_idx in self.neighbors[u_idx]:
                alt = dist[u_idx] + np.linalg.norm(self.nodes[u_idx].position[:2] -
                                                   self.nodes[v_idx].position[:2])
                if alt < dist[v_idx]:
                    dist[v_idx] = alt
                    prev[v_idx] = u_idx
                    heapq.heappush(pq, (alt, v_idx))

        # Reconstruct path from end to start
        path_node_position = []
        path_node_idx = []
        u = end
        u_idx = u.idx
        while u_idx != -1:
            path_node_position.append(self.nodes[u_idx].position)
            path_node_idx.append(u_idx)
            u_idx = prev[u_idx]
        path_node_position.reverse()
        path_node_idx.reverse()

        # If the path doesn't reach the start node, return an empty path
        if tuple(path_node_position[0]) != tuple(start.position):
            return np.array([]), np.array([])
        return np.array(path_node_position), path_node_idx

    def check_connected(self, start, end):
        # 将输入点映射到图中最近的节点
        start = self.nodes[start]
        end = self.nodes[end]

        start_idx = start.idx
        end_idx = end.idx

        # 初始化距离数组
        dist = np.full(self.node_cnt, np.inf)
        dist[start_idx] = 0

        # Dijkstra 使用的优先队列
        pq = []
        heapq.heappush(pq, (0, start_idx))

        while pq:
            current_dist, u_idx = heapq.heappop(pq)

            # 如果已经到达目标节点，说明 start 与 end 相通
            if u_idx == end_idx:
                return True

            # 如果不是最短距离，跳过
            if current_dist > dist[u_idx]:
                continue

            # 遍历相邻节点
            for v_idx in self.neighbors[u_idx]:
                alt = dist[u_idx] + np.linalg.norm(
                    self.nodes[u_idx].position[:2] - self.nodes[v_idx].position[:2]
                )
                if alt < dist[v_idx]:
                    dist[v_idx] = alt
                    heapq.heappush(pq, (alt, v_idx))

        # 如果最终没有到达 end 节点，说明不相通
        return False

    def update_edges(self):
        cluster_points = self.navigable_pcd.point.positions.cpu().numpy().copy()
        obstacle_points = self.obstacle_pcd.point.positions.cpu().numpy().copy()

        nav_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        obstacle_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        cluster_points_map = translate_point_to_grid(cluster_points, self.grid_resolution, self.voxel_dimension)[:, :2]
        obstacle_points_map = translate_point_to_grid(obstacle_points, self.grid_resolution, self.voxel_dimension)[:, :2]

        nav_map[cluster_points_map[:, 0], cluster_points_map[:, 1]] = 1
        obstacle_map[obstacle_points_map[:, 0], obstacle_points_map[:, 1]] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        obstacle_map = cv2.dilate(obstacle_map, kernel, iterations=1)
        obstacle_map = cv2.bitwise_not(obstacle_map)
        nav_map = cv2.bitwise_and(nav_map, obstacle_map)

        # get all the possible 2 nodes pair
        node_pairs = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_pairs.append((i, j))

        for i, j in node_pairs:
            node_i = self.nodes[i]
            node_j = self.nodes[j]

            point1 = node_i.position.copy()
            point2 = node_j.position.copy()

            # find the closet point in cluster_points to point1 and point2
            distance1 = np.linalg.norm(cluster_points[:, :2] - point1[:2], axis=1)
            distance2 = np.linalg.norm(cluster_points[:, :2] - point2[:2], axis=1)

            point1 = cluster_points[np.argmin(distance1)]
            point2 = cluster_points[np.argmin(distance2)]

            point1_map = translate_point_to_grid(point1, self.grid_resolution, self.voxel_dimension)[:2]
            point2_map = translate_point_to_grid(point2, self.grid_resolution, self.voxel_dimension)[:2]

            interpolate_points = bresenham(point1_map[0], point1_map[1], point2_map[0], point2_map[1])
            interpolate_points = list(interpolate_points)[1:-1]

            obs_num = 0
            for point in interpolate_points:
                if nav_map[point[0], point[1]] == 0:
                    obs_num = obs_num + 1

            if obs_num > 5:
                if node_i.idx in self.neighbors[node_j.idx]:
                    self.remove_edge(node_i.idx, node_j.idx)
                    if not self.check_connected(node_i.idx, node_j.idx):
                        self.add_edge(node_i.idx, node_j.idx)
            else:
                if node_i.idx not in self.neighbors[node_j.idx]:
                    self.add_edge(node_i.idx, node_j.idx)
        edges = self.get_edges()
        centers = [node.position for node in self.nodes]
        # apply the average z
        centers = np.array(centers)
        centers[:, 2] = np.ones_like(centers[:, 2]) * self.floor_height

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(centers)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color([1, 0, 0])
        os.makedirs(f'{self.save_dir}/episode-{self.episode_idx}/edges/', exist_ok=True)
        o3d.io.write_line_set(
            f'{self.save_dir}/episode-{self.episode_idx}/edges/edges_{self.update_iterations}.ply',
            line_set)

    def update(self,
               rgb,
               depth,
               position,
               rotation,
               episode_idx,
               episode_step,
               on_node_flag,
               current_node_idx):

        self.episode_idx = episode_idx

        self.current_position = self.translation_func(position) - self.initial_position
        self.current_node_idx = current_node_idx
        self.rotation = rotation
        self.current_rotation = self.rotation_func(rotation)
        self.original_depth = depth.copy()
        self.ori_current_depth = ori_preprocess_depth(depth) # 直接每个pixel都根据max,min做clip
        self.current_depth = preprocess_depth(depth, self.lower_img, self.upper_img) # 球面深度clip
        self.current_rgb = preprocess_image(rgb)

        self.trajectory_position.append(self.current_position)

        start_o = math.degrees((np.arctan2(-self.current_rotation[1, 2],
                                           -self.current_rotation[0, 2])))
        self.traj_on_trav.append([self.current_position[0], self.current_position[1], start_o])

        # to avoid there is no valid depth value (especially in real-world)
        if np.sum(self.ori_current_depth) > 0:
            camera_points, camera_colors = get_pointcloud_from_depth(self.current_rgb,
                                                                     self.ori_current_depth,
                                                                     self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position,
                                              self.current_rotation)
            self.current_pcd = gpu_pointcloud_from_array(
                world_points, camera_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
        else:
            # create 2 empty pointclouds
            self.current_pcd = None
            self.current_navigable_pcd = None
            return

        # pointcloud update
        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd,
                                              self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        self.scene_pcd = self.scene_pcd.select_by_index((self.scene_pcd.point.positions[:, 2]
                                                         > self.floor_height - 0.2).nonzero()[0])
        self.useful_pcd = self.scene_pcd.select_by_index((self.scene_pcd.point.positions[:, 2]
                                                          < self.ceiling_height).nonzero()[0])


        # geometry
        current_navigable_point = self.current_pcd.select_by_index(
            (self.current_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0])
        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        max_distance = 1.8

        standing_position = np.array(
            [self.current_position[0], self.current_position[1], self.floor_height])

        whole_pcd = self.useful_pcd.select_by_index((self.useful_pcd.point.positions[:, 2]
                                                     < self.ceiling_height).nonzero()[0])
        whole_pcd = whole_pcd.select_by_index((whole_pcd.point.positions[:, 2]
                                               > self.floor_height + 0.2).nonzero()[0])
        whole_pcd_points = whole_pcd.point.positions.cpu().numpy()
        whole_pcd_points[:, 2] = np.ones_like(whole_pcd_points[:, 2]) * self.floor_height
        whole_pcd = gpu_pointcloud_from_array(whole_pcd_points,
                                              np.ones_like(whole_pcd_points) * 100, self.pcd_device)
        whole_pcd = whole_pcd.voxel_down_sample(self.pcd_resolution)
        whole_pcd_points = whole_pcd.point.positions.cpu().numpy()

        current_pcd_points = self.current_pcd.point.positions.cpu().numpy()
        current_pcd_points[:, 2] = np.ones_like(current_pcd_points[:, 2]) * self.floor_height
        current_pcd = gpu_pointcloud_from_array(current_pcd_points,
                                                np.ones_like(current_pcd_points) * 100,
                                                self.pcd_device)
        current_pcd = current_pcd.voxel_down_sample(self.pcd_resolution)
        current_pcd_points = current_pcd.point.positions.cpu().numpy()

        closest_distances, closest_points = self.get_closest_disances_and_points(
            current_pcd_points, whole_pcd_points, standing_position, max_distance)

        if closest_points.shape[0] != 0:
            # interpolate the nearby blind points
            interpolate_points = np.linspace(
                np.ones_like(closest_points) * standing_position, closest_points,
                60).reshape(-1, 3)
            interpolate_points = interpolate_points[
                (interpolate_points[:, 2] > self.floor_height - 0.2) &
                (interpolate_points[:, 2] < self.floor_height + 0.2)]
            interpolate_points = np.concatenate((current_navigable_position, interpolate_points),
                                                axis=0)

            interpolate_points[:, 2] = np.ones_like(interpolate_points[:, 2]) * self.floor_height

            traversable_points = []
            # use self.env.sim.is_navigable to check the navigability of interpolate_points
            for point in interpolate_points:
                point_sim = point + self.initial_position
                point_sim = [point_sim[0], point_sim[2], point_sim[1]]
                if self.env.sim.is_navigable(point_sim):
                    traversable_points.append(point)
            traversable_points = np.array(traversable_points)

            interpolate_colors = np.ones_like(interpolate_points) * 100
            self.current_navigable_pcd = gpu_pointcloud_from_array(interpolate_points,
                                                                   interpolate_colors,
                                                                   self.pcd_device)

            self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,
                                                      self.current_navigable_pcd).voxel_down_sample(
                                                          self.pcd_resolution)
        else:
            self.current_navigable_pcd = self.current_pcd.select_by_index(
                (self.current_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0])
            current_navigable_pcd_position = self.current_navigable_pcd.point.positions.cpu().numpy(
            )
            current_navigable_pcd_position[:, 2] = np.ones_like(
                current_navigable_pcd_position[:, 2]) * self.floor_height
            traversable_points = []
            # use self.env.sim.is_navigable to check the navigability of interpolate_points
            for point in current_navigable_pcd_position:
                point_sim = point + self.initial_position
                point_sim = [point_sim[0], point_sim[2], point_sim[1]]
                if self.env.sim.is_navigable(point_sim):
                    traversable_points.append(point)
            traversable_points = np.array(traversable_points)

            self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,
                                                      self.current_navigable_pcd)
            if not self.navigable_pcd.is_empty(): # downsample the pointcloud
                self.navigable_pcd = self.navigable_pcd.voxel_down_sample(self.pcd_resolution)

        traversable_colors = np.ones_like(traversable_points) * 100
        self.current_traversable_pcd = gpu_pointcloud_from_array(traversable_points,
                                                                 traversable_colors,
                                                                 self.pcd_device)
        self.traversable_pcd_all = gpu_merge_pointcloud(self.traversable_pcd_all,
                                                    self.current_traversable_pcd)
        self.traversable_pcd = self.traversable_pcd_all.clone()

        if not self.traversable_pcd.is_empty():
            self.traversable_pcd_all = self.traversable_pcd_all.voxel_down_sample(self.pcd_resolution)
            self.traversable_pcd = self.traversable_pcd.voxel_down_sample(self.pcd_resolution)

        if not on_node_flag:
            # update th process obs_pcd
            self.process_nav_pcd = gpu_merge_pointcloud(self.process_nav_pcd,
                                                        self.current_navigable_pcd)
            try:
                process_nav_pcd_position = self.process_nav_pcd.point.positions.cpu().numpy()
                process_nav_pcd_position[:, 2] = np.ones_like(
                    process_nav_pcd_position[:, 2]) * np.mean(process_nav_pcd_position[:, 2])
                self.process_nav_pcd.point["positions"] = o3d.core.Tensor(
                    process_nav_pcd_position, dtype=o3d.core.Dtype.Float32, device=self.pcd_device)
                self.process_nav_pcd = self.process_nav_pcd.voxel_down_sample(self.pcd_resolution)
            except:
                pass

        if not on_node_flag:
            # update th process obs_pcd
            self.process_obs_pcd = gpu_merge_pointcloud(self.process_obs_pcd, self.current_pcd)
            self.process_obs_pcd = gpu_merge_pointcloud(
                self.process_obs_pcd,
                self.current_navigable_pcd).voxel_down_sample(self.pcd_resolution)
            try:
                self.process_obs_pcd = self.process_obs_pcd.select_by_index(
                    (self.process_obs_pcd.point.positions[:, 2]
                     > self.floor_height - 0.2).nonzero()[0])
                self.process_obs_pcd = self.process_obs_pcd.select_by_index(
                    (self.process_obs_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])
            except:
                pass

        # filter the obstacle pointcloud
        self.obstacle_pcd = self.useful_pcd.select_by_index(
            (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.2).nonzero()[0])
        self.trajectory_pcd = gpu_pointcloud_from_array(
            np.array(self.trajectory_position), np.zeros((len(self.trajectory_position), 3)),
            self.pcd_device)

        self.update_iterations += 1

    def update_trav_map(self, navigable_pcd):
        nav_points = navigable_pcd.point.positions.cpu().numpy()[:, :2]

        # print(len(nav_points))

        nav_points_index = np.floor((nav_points + self.map_size * self.map_resolution / 2.0) /
                                    self.map_resolution).astype(int)

        # print(len(nav_points_index))
        trav_map_np = self.trav_map.cpu().numpy()
        trav_map_np[nav_points_index[:, 0], nav_points_index[:, 1]] = 1

        trav_map_np = 1 - trav_map_np

        selem = skimage.morphology.disk(2)
        trav_map_np = skimage.morphology.binary_dilation(trav_map_np, selem).astype(np.float32)

        # get the mask of self.collison_map
        collision_mask = self.collision_map > 0
        trav_map_np[collision_mask] = 1

        trav_map_np = skimage.morphology.binary_dilation(trav_map_np, selem) != True
        # trav_map_np = 1 - trav_map_np

        trav_map_np = trav_map_np * 1.0

        self.trav_map = torch.from_numpy(trav_map_np).float().to(self.device)

        # tmp_vector = [0.0, 0.0, -1.0]
        # world_tmp_vector = self.current_rotation @ tmp_vector
        # angle = math.atan2(world_tmp_vector[1], world_tmp_vector[0]) * 180 / math.pi
        # length = 80
        # print(f'angle2: {angle}')
        #
        # center = [self.map_size // 2, self.map_size // 2]
        #
        # for i in range(length):
        #     x_ = int(center[0] + i * math.cos(math.radians(angle)))
        #     y_ = int(center[1] + i * math.sin(math.radians(angle)))
        #     self.trav_map[x_, y_] = 0.5

        # save the traversability map as an image and draw the trajectory on it
        trav_map_to_save = self.trav_map.cpu().numpy()
        for traj_x, traj_y, ori in self.traj_on_trav:
            x = int((traj_x + self.map_size * self.map_resolution / 2.0) / self.map_resolution)
            y = int((traj_y + self.map_size * self.map_resolution / 2.0) / self.map_resolution)
            trav_map_to_save[x, y] = 0.5

        # traj_x, traj_y, ori = self.traj_on_trav[-1]
        # for i in range(80):
        #     x = int((traj_x + self.map_size * self.map_resolution / 2.0) / self.map_resolution)
        #     y = int((traj_y + self.map_size * self.map_resolution / 2.0) / self.map_resolution)
        #     x_ = int(x + i * math.cos(math.radians(ori)))
        #     y_ = int(y + i * math.sin(math.radians(ori)))
        #     trav_map_to_save[x_, y_] = 0.5

        trav_map_img = trav_map_to_save
        # switch the x and y axis
        trav_map_img = np.transpose(trav_map_img)
        # flip the y axis
        trav_map_img = np.flip(trav_map_img, axis=0)
        trav_map_img = (trav_map_img * 255).astype(np.uint8)
        trav_map_img = cv2.resize(trav_map_img, (960, 960), interpolation=cv2.INTER_NEAREST)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_idx}/progress_info/', exist_ok=True)
        cv2.imwrite(
            f'{self.save_dir}/episode-{self.episode_idx}/progress_info/trav_map_{self.update_iterations}.png',
            trav_map_img)

        # save the collision map as an image
        collision_map_img = self.collision_map
        # switch the x and y axis
        collision_map_img = np.transpose(collision_map_img)
        # flip the y axis
        collision_map_img = np.flip(collision_map_img, axis=0)
        collision_map_img = (collision_map_img * 255).astype(np.uint8)
        collision_map_img = cv2.resize(collision_map_img, (960, 960),
                                       interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{self.save_dir}/episode-{self.episode_idx}/collision_map.png',
                    collision_map_img)

    def update_object_pcd(self):
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.objects:
            points = entity.pcd.point.positions.cpu().numpy()
            colors = entity.pcd.point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        try:
            return gpu_pointcloud(object_pcd, self.pcd_device)
        except:
            return self.scene_pcd

    def get_view_pointcloud(self, rgb, depth, translation, rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = ori_current_depth(depth, self.lower_img, self.upper_img)
        current_rgb = preprocess_image(rgb)
        camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth,
                                                                 self.camera_intrinsic)
        world_points = translate_to_world(camera_points, current_position, current_rotation)
        current_pcd = gpu_pointcloud_from_array(
            world_points, camera_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
        return current_pcd

    def get_object_entities(self, ori_depth, depth, rgb, classes, boxes, masks, confidences):
        entities = []
        for ent in self.objects:
            if ent.tag not in self.exist_objects:
                self.exist_objects.append(ent.tag)
        for cls, bbox, mask, score in zip(classes, boxes, masks, confidences):
            # if depth[mask > 0].min() < 1.0 and score < 0.5:
            #     logger.info("Warning: Too close and low confidence")
            #     continue
            if cls not in self.exist_objects:
                self.exist_objects.append(cls)

            camera_points = get_pointcloud_from_depth_mask(ori_depth, mask, self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position,
                                              self.current_rotation)
            point_colors = np.array([d3_40_colors_rgb[self.exist_objects.index(cls) % 40]] *
                                    world_points.shape[0])
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(
                world_points, point_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
            object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                continue

            tmp_camera_points = get_pointcloud_from_depth_mask(depth, mask, self.camera_intrinsic)
            tmp_world_points = translate_to_world(tmp_camera_points, self.current_position,
                                                  self.current_rotation)
            if tmp_world_points.shape[0] < 10:
                continue
            tmp_point_colors = np.array([d3_40_colors_rgb[self.exist_objects.index(cls) % 40]] *
                                        tmp_world_points.shape[0])
            tmp_object_pcd = gpu_pointcloud_from_array(tmp_world_points, tmp_point_colors,
                                                       self.pcd_device).voxel_down_sample(
                                                           self.pcd_resolution)
            tmp_object_pcd = gpu_cluster_filter(tmp_object_pcd)
            actual_num = tmp_object_pcd.point.positions.shape[0]
            if actual_num < 10:
                continue

            xyzpos = np.mean(object_pcd.point.positions.cpu().numpy(), axis=0)
            entity = ObjectNode(object_pcd, cls, score, xyzpos, actual_num, rgb, bbox)
            entities.append(entity)
        return entities

    def get_object_entities_pano(self, depth, rgb, poss, rots, classes, boxes, masks, confidences,
                                 depth_list):
        entities = []
        for ent in self.objects:
            if ent.tag not in self.exist_objects:
                self.exist_objects.append(ent.tag)
        for cls, bbox, mask, score in zip(classes, boxes, masks, confidences):
            if cls not in self.exist_objects:
                self.exist_objects.append(cls)

            world_points = []
            tmp_world_points = []
            idx = []
            for i in range(3):
                wmin = 640 * i
                wmax = 640 * (i + 1)
                nwbox = bbox.clone()
                if nwbox[2] < wmin or nwbox[0] >= wmax:
                    continue
                nwbox[0] = max(nwbox[0], wmin)
                nwbox[2] = min(nwbox[2], wmax)
                camera_points = get_pointcloud_from_depth_mask(
                    depth[:, wmin:wmax], mask[:, wmin:wmax].numpy(), self.camera_intrinsic,
                    self.coord_maps[i], depth_list[i])
                nw_points = translate_to_world(camera_points, poss[i], rots[i])
                if nw_points.shape[0] == 0:
                    continue
                world_points.append(nw_points)
                idx.append(i)
                tmp_depth = depth[:, wmin:wmax].copy()
                tmp_depth = preprocess_depth(tmp_depth, 0.51, 4.99)
                tmp_depth_list = depth_list[i].copy()
                tmp_depth_list = preprocess_depth(tmp_depth_list, 0.51, 4.99)
                tmp_camera_points = get_pointcloud_from_depth_mask(
                    tmp_depth, mask[:, wmin:wmax].numpy(), self.camera_intrinsic,
                    self.coord_maps[i], tmp_depth_list)
                tmp_points = translate_to_world(tmp_camera_points, poss[i], rots[i])
                tmp_world_points.append(tmp_points)

            if len(world_points) == 0:
                continue

            # clrs = []
            # for i in range(len(world_points)):
                # clrs.append(np.array([d3_40_colors_rgb[idx[i]]] * world_points[i].shape[0]))
            world_points = np.concatenate(world_points, axis=0)
            point_colors = np.array([d3_40_colors_rgb[self.exist_objects.index(cls) % 40]] *
                                    world_points.shape[0])
            # point_colors = np.concatenate(clrs, axis=0)
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(
                world_points, point_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
            # object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                continue

            tmp_world_points = np.concatenate(tmp_world_points, axis=0)
            if tmp_world_points.shape[0] < 10:
                continue
            tmp_point_colors = np.array([d3_40_colors_rgb[self.exist_objects.index(cls) % 40]] *
                                        tmp_world_points.shape[0])
            tmp_object_pcd = gpu_pointcloud_from_array(tmp_world_points, tmp_point_colors,
                                                       self.pcd_device).voxel_down_sample(
                                                           self.pcd_resolution)
            tmp_object_pcd = gpu_cluster_filter(tmp_object_pcd)
            actual_num = tmp_object_pcd.point.positions.shape[0]
            if actual_num < 10:
                continue

            # xyzpos = np.mean(object_pcd.point.positions.cpu().numpy(), axis=0)
            xyzpos = np.mean(tmp_object_pcd.point.positions.cpu().numpy(), axis=0)
            entity = ObjectNode(object_pcd, tmp_object_pcd, cls, score, xyzpos, actual_num, rgb, bbox)
            entities.append(entity)
        return entities

    def associate_object_entities(self, ref_entities, eval_entities):
        entity_indices = []
        for entity in eval_entities:
            if len(ref_entities) == 0:
                ref_entities.append(entity)
                entity_indices.append(0)
                continue
            overlap_scores = []
            overlap_scores1 = []
            overlap_scores2 = []
            eval_pcd = entity.pcd
            for ref_entity in ref_entities:
                if eval_pcd.point.positions.shape[0] == 0:
                    break
                if ref_entity.alive == False:
                    continue
                cdist1 = pointcloud_distance(eval_pcd, ref_entity.pcd)
                cdist2 = pointcloud_distance(ref_entity.pcd, eval_pcd)
                cdist_all = torch.cat([cdist1, cdist2], dim=0)
                overlap_condition1 = (cdist1 < 0.1)
                overlap_condition2 = (cdist2 < 0.1)
                overlap_condition_all = (cdist_all < 0.1)
                # nonoverlap_condition = overlap_condition.logical_not()
                # eval_pcd = eval_pcd.select_by_index(
                #     o3d.core.Tensor(nonoverlap_condition.cpu().numpy(),
                #                     device=self.pcd_device).nonzero()[0])
                overlap_scores1.append(
                    (overlap_condition1.sum() / (overlap_condition1.shape[0] + 1e-6)).cpu().numpy())
                overlap_scores2.append(
                    (overlap_condition2.sum() / (overlap_condition2.shape[0] + 1e-6)).cpu().numpy())
                overlap_scores.append(
                    (overlap_condition_all.sum() / (overlap_condition_all.shape[0] + 1e-6)).cpu().numpy())
            max_overlap_score = np.max(overlap_scores)
            arg_overlap_index = np.argmax(overlap_scores)


            if max_overlap_score < 0.25:
                entity.pcd = eval_pcd
                ref_entities.append(entity)
                entity_indices.append(len(ref_entities) - 1)
            else:
                # target class
                if entity.tag == self.target:
                    argmax_entity = ref_entities[arg_overlap_index]
                    print(argmax_entity.tag)
                    if argmax_entity.tag == 'nothing':
                        continue
                    else:
                        overlap_score = overlap_scores[arg_overlap_index]
                        overlap_score1 = overlap_scores1[arg_overlap_index]
                        overlap_score2 = overlap_scores2[arg_overlap_index]
                        # check if this 2 is the same object:
                        if overlap_score > 0.85 and overlap_score1 > 0.85 and overlap_score2 > 0.85:
                            argmax_entity.num_list.pop(argmax_entity.tag)
                            argmax_entity.conf_list.pop(argmax_entity.tag)
                            argmax_entity.tag = self.target
                            argmax_entity.pcd = eval_pcd
                            argmax_entity.position = entity.position
                            argmax_entity.confidence = entity.confidence
                            argmax_entity.num_list[self.target] = entity.num_list[self.target]
                            argmax_entity.conf_list[self.target] = entity.conf_list[self.target]
                        else:
                            entity.pcd = eval_pcd
                            ref_entities.append(entity)
                            entity_indices.append(len(ref_entities) - 1)
                else:
                    argmax_entity = ref_entities[arg_overlap_index]
                    argmax_entity.pcd = gpu_merge_pointcloud(argmax_entity.pcd, eval_pcd)
                    argmax_entity.position = np.mean(argmax_entity.pcd.point.positions.cpu().numpy(),
                                                     axis=0)
                    for node in entity.nodes:
                        if node not in argmax_entity.nodes:
                            argmax_entity.nodes.append(node)

                    alpha = 2
                    ori_tag = argmax_entity.tag
                    new_tag = entity.tag

                    if new_tag not in argmax_entity.num_list.keys():
                        argmax_entity.num_list[new_tag] = entity.num_list[new_tag]
                        argmax_entity.conf_list[new_tag] = entity.conf_list[new_tag]
                    else:
                        n1 = argmax_entity.num_list[new_tag]
                        n2 = entity.num_list[new_tag]
                        c1 = argmax_entity.conf_list[new_tag]
                        c2 = entity.conf_list[new_tag]
                        w1 = (c1**alpha * n1) / (c1**alpha * n1 + c2**alpha * n2)
                        w2 = 1 - w1
                        argmax_entity.conf_list[new_tag] = c1 * w1 + c2 * w2
                        argmax_entity.num_list[new_tag] = n1 + n2

                    if new_tag == ori_tag:
                        argmax_entity.tag = new_tag
                        argmax_entity.confidence = argmax_entity.conf_list[new_tag]
                    else:
                        n1 = argmax_entity.num_list[ori_tag]
                        n2 = argmax_entity.num_list[new_tag]
                        c1 = argmax_entity.conf_list[ori_tag]
                        c2 = argmax_entity.conf_list[new_tag]
                        w1 = (c1**alpha * n1) / (c1**alpha * n1 + c2**alpha * n2)
                        w2 = 1 - w1
                        if w1 > w2:
                            argmax_entity.tag = ori_tag
                            argmax_entity.confidence = c1
                        else:
                            argmax_entity.tag = new_tag
                            argmax_entity.confidence = c2

                    # if argmax_entity.pcd.point.positions.shape[0] < \
                    #    entity.pcd.point.positions.shape[0] or entity.tag in INTEREST_OBJECTS:
                    #     if entity.confidence > argmax_entity.confidence:
                    #         argmax_entity.tag = entity.tag
                    #         argmax_entity.confidence = entity.confidence

                    ref_entities[arg_overlap_index] = argmax_entity
                    entity_indices.append(arg_overlap_index)
        entity_indices = [int(i) for i in entity_indices]
        return ref_entities, entity_indices

    def get_appeared_objects(self):
        return [entity.tag for entity in self.objects]

    def save_pointcloud_debug(self, path="./"):

        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(
                self.useful_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply", save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(
                self.navigable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(
                self.navigable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply", save_pcd)
        except:
            pass
        try:
            assert self.obstacle_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(
                self.obstacle_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(
                self.obstacle_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "obstacle.ply", save_pcd)
        except:
            pass
        try:
            assert self.traversable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(
                self.traversable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(
                self.traversable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "traversable.ply", save_pcd)
        except:
            pass
        try:
            assert self.traversable_pcd_all.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(
                self.traversable_pcd_all.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(
                self.traversable_pcd_all.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "traversable_all.ply", save_pcd)
        except:
            pass

        object_pcd = o3d.geometry.PointCloud()
        for entity in self.objects:
            points = entity.pcd.point.positions.cpu().numpy()
            colors = entity.pcd.point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply", object_pcd)

        # save the nodes
        center_spheres = []
        centers = self.get_nodes_positions()
        for center in centers:
            center[2] = self.floor_height
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(center)
            center_spheres.append(sphere)
        # save the center_spheres
        combined_sphere = o3d.geometry.TriangleMesh()
        for sphere in center_spheres:
            combined_sphere += sphere
        o3d.io.write_triangle_mesh(path + "center_spheres.ply", combined_sphere)

        # save nodes with object
        nwpcd = object_pcd
        combined_sphere_pcd = o3d.geometry.PointCloud()
        combined_sphere_pcd.points = combined_sphere.vertices
        combined_sphere_pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(combined_sphere.vertices) * np.array([1, 0, 0]))

        nwpcd = nwpcd + combined_sphere_pcd
        o3d.io.write_point_cloud(path + "combined_scene.ply", nwpcd)

        obj_centers = []
        for i in range(len(self.objects)):
            entity = self.objects[i]
            points = entity.pcd.point.positions.cpu().numpy()
            pos = np.mean(points, axis=0)
            obj_centers.append(pos)
        num_obj = len(obj_centers)

        for i in range(self.node_cnt):
            center = centers[i]
            center[2] = self.floor_height
            obj_centers.append(center)

        obj_centers = np.array(obj_centers)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(obj_centers)

        edges = []

        for i in range(self.node_cnt):
            node = self.nodes[i]
            center = node.position
            logger.info("Node: {}, Center: {}", i, center)
            for ind in node.objects:
                obj = self.objects[ind]
                logger.info("Object: {}", obj.tag)
                edges.append([ind, i + num_obj])

        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(np.ones((len(edges), 3)) * np.array([1, 0, 1]))

        # visualize the edges with nwpcd
        o3d.io.write_line_set(path + "edges.ply", lineset)

        # save the self.nodes in the txt file
        np.savetxt(path + "nodes.txt", np.array(centers), fmt='%f')

        edges = self.get_edges()

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(centers)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color([1, 0, 0])

        o3d.io.write_line_set(path + "graph.ply", line_set)

    def calculate_intersections(self, point_cloud, current_position, num_rays=72, max_distance=2.5):
        point_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        point_positions = point_cloud
        point_grid_idxs = translate_point_to_grid(point_positions, self.grid_resolution, self.voxel_dimension)
        point_map[point_grid_idxs[:, 0], point_grid_idxs[:, 1]] = 1

        intersections = []
        intersections_false_postive = []
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        origin_point = translate_point_to_grid(current_position, self.grid_resolution, self.voxel_dimension)[:2]
        max_distance_map = int(max_distance / self.grid_resolution)

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            end_point = origin_point + direction * max_distance_map

            # print(origin_point, end_point)

            interpolate_points = bresenham(origin_point[0], origin_point[1], int(end_point[0]), int(end_point[1]))
            # output the interpolate_points
            interpolate_points = list(interpolate_points)
            # print(interpolate_points)
            for idx, point in enumerate(interpolate_points):
                if point_map[point[0], point[1]] == 0:
                    point = np.array(list(point))
                    point_1 = translate_single_grid_to_point(point, self.grid_resolution, self.voxel_dimension)[:2] + 0.15 * direction
                    intersections.append(point_1)
                    intersections_false_postive.append(point_1)
                    break
                if idx == len((interpolate_points)) - 1:
                    point = np.array(list(point))
                    point_1 = translate_single_grid_to_point(point, self.grid_resolution, self.voxel_dimension)[:2] + 0.15 * direction
                    intersections_false_postive.append(point_1)

            # for idx, d in enumerate(np.linspace(0, max_distance, delta)):  # 逐渐增大距离
            #     point = current_position + d * direction
            #
            #     if self.is_out_of_boundary(point_cloud, point):
            #         # if self.is_out_of_boundary(point_cloud, point) or idx == delta - 1:
            #         point = current_position + (d + 0.15) * direction
            #         intersections.append(point)
            #         intersections_false_postive.append(point)
            #         break
            #     if idx == delta - 1:
            #         intersections_false_postive.append(point)

        intersections = np.array(intersections)
        intersections_false_postive = np.array(intersections_false_postive)

        if intersections.shape[0] == 0:
            distance_inter = np.linalg.norm(intersections_false_postive - current_position[:2], axis=1)
            intersections_false_postive = intersections_false_postive[distance_inter > 0.2]
            return intersections_false_postive

        distance_inter = np.linalg.norm(intersections - current_position[:2], axis=1)
        intersections = intersections[distance_inter > 0.2]
        polygon_path = Path(intersections)
        # Remove points inside the polygon
        if polygon_path.contains_point(current_position[:2]):
            return intersections
        else:
            distance_inter = np.linalg.norm(intersections_false_postive - current_position[:2], axis=1)
            intersections_false_postive = intersections_false_postive[distance_inter > 0.2]
            return intersections_false_postive

    def save_point_cloud(self, waypoint_gpu_pcd, obs_gpu_pcd, frontier, intersection, nav_gpu_pcd,
                         current_traversable_pcd, valid_centers, episode_idx, step_idx):
        room_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        for room_node in self.room_nodes:
            room_mask = room_node.mask_map
            room_idx = room_node.room_id
            room_map[room_mask[:,0], room_mask[:,1]] = room_idx + 1
        color_image = save_grid_map(room_map)
        color_map = {
            # 0: [0, 0, 0],  # Black
            1: [255, 0, 0],  # Blue
            2: [0, 255, 0]  # Green
        }
        # save the frontier_map(np array) as a image
        os.makedirs(f'{self.save_dir}/episode-{episode_idx}/frontier_map', exist_ok=True)
        # switch the x and y axis
        frontier_map_save = np.transpose(self.grid_map)
        # flip the y axis
        frontier_map_save = np.flip(frontier_map_save, axis=0)

        for val, color in color_map.items():
            color_image[frontier_map_save == val] = color
        cv2.imwrite(f'{self.save_dir}/episode-{episode_idx}/frontier_map/frontier_map_{step_idx}.png',
                    color_image)
        frontier_map_save = np.transpose(self.frontiers_considered)
        # flip the y axis
        frontier_map_save = np.flip(frontier_map_save, axis=0)
        cv2.imwrite(f'{self.save_dir}/episode-{episode_idx}/frontier_map/frontier_map_considered_{step_idx}.png',
                    frontier_map_save*255)

        # # save the waypoint
        # try:
        #     waypoint_pcd = o3d.geometry.PointCloud()
        #     waypoint_pcd.points = o3d.utility.Vector3dVector(
        #         waypoint_gpu_pcd.point.positions.cpu().numpy())
        #     waypoint_pcd.paint_uniform_color([0, 1, 0])
        #     os.makedirs(f'{self.save_dir}/episode-{episode_idx}/waypoint', exist_ok=True)
        #     o3d.io.write_point_cloud(
        #         f'{self.save_dir}/episode-{episode_idx}/waypoint/waypoint_{step_idx}.ply',
        #         waypoint_pcd)
        # except:
        #     logger.info('Failed to save waypoint')
        #     pass

        # save the obs
        try:
            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(obs_gpu_pcd.point.positions.cpu().numpy())
            obs_pcd.paint_uniform_color([0, 0, 0])
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/process_obs', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/process_obs/obs_{step_idx}.ply', obs_pcd)
        except:
            logger.info('Failed to save obs')
            pass

        # save the frontier
        try:
            frontier_pcd = o3d.geometry.PointCloud()
            frontier_pcd.points = o3d.utility.Vector3dVector(frontier)
            frontier_pcd.paint_uniform_color([0, 1, 0])
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/frontier', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/frontier/frontier_{step_idx}.ply',
                frontier_pcd)
        except:
            logger.info('Failed to save frontier')
            pass

        # save the intersection
        try:
            intersection_pcd = o3d.geometry.PointCloud()
            intersection_pcd.points = o3d.utility.Vector3dVector(intersection)
            intersection_pcd.paint_uniform_color([0, 1, 0])
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/intersection', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/intersection/intersection_{step_idx}.ply',
                intersection_pcd)
        except:
            logger.info('Failed to save intersection')
            pass

        # save the nav

        try:
            nav_pcd = o3d.geometry.PointCloud()
            nav_pcd.points = o3d.utility.Vector3dVector(nav_gpu_pcd.point.positions.cpu().numpy())
            nav_pcd.colors = o3d.utility.Vector3dVector(nav_gpu_pcd.point.colors.cpu().numpy())
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/nav', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/nav/nav_{step_idx}.ply', nav_pcd)
        except:
            logger.info('Failed to save nav')
            pass

        try:
            traversable_pcd = o3d.geometry.PointCloud()
            traversable_pcd.points = o3d.utility.Vector3dVector(
                current_traversable_pcd.point.positions.cpu().numpy())
            traversable_pcd.colors = o3d.utility.Vector3dVector(
                current_traversable_pcd.point.colors.cpu().numpy())
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/traversable', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/traversable/traversable_{step_idx}.ply',
                traversable_pcd)
        except:
            logger.info('Failed to save traversable')
            pass

        assert self.node_cnt == len(self.nodes)
        current_node = self.nodes[self.current_node_idx]
        current_node_idx = current_node.idx
        if self.node_cnt > 0:
            center_spheres = []
            center_labels = []
            for center_idx, node in enumerate(self.nodes):
                center = node.position
                center[2] = self.floor_height
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                if node.idx==current_node_idx:
                    sphere.paint_uniform_color([0, 0, 1])
                elif node.state == 1:
                    sphere.paint_uniform_color([0, 0, 0])
                elif node.has_frontier:
                    if node.has_true_frontier:
                        sphere.paint_uniform_color([0, 1, 0])
                    else:
                        sphere.paint_uniform_color([0, 0.25, 0])
                else:
                    sphere.paint_uniform_color([1, 0, 0])
                sphere.translate(center)
                center_spheres.append(sphere)

                number_lable = create_number_point_cloud(node.idx, center)
                center_labels.append(number_lable)

            # save the center_spheres
            combined_sphere = o3d.geometry.TriangleMesh()
            for sphere in center_spheres:
                combined_sphere += sphere

            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/centers', exist_ok=True)
            o3d.io.write_triangle_mesh(
                f'{self.save_dir}/episode-{episode_idx}/centers/centers_{step_idx}.ply',
                combined_sphere)

            # save the center_labels
            combined_labels = o3d.geometry.PointCloud()
            for label in center_labels:
                combined_labels += label
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/centers', exist_ok=True)
            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{episode_idx}/centers/centers_label_{step_idx}.ply',
                combined_labels)

        try:
            # save the room as ply
            rooms = o3d.geometry.PointCloud()
            for room_node in self.room_nodes:
                room_positions = room_node.mask
                room_pcd = o3d.geometry.PointCloud()
                room_pcd.points = o3d.utility.Vector3dVector(room_positions)
                # paint a random color
                room_pcd.paint_uniform_color(np.random.rand(3))
                rooms += room_pcd
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/room', exist_ok=True)
            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/room/room_{step_idx}.ply', rooms)
        except:
            logger.info('Failed to save room')

        try:
            # save a whole visualization
            room_spheres = o3d.geometry.TriangleMesh()
            room_labels = o3d.geometry.PointCloud()
            room_viewpoint_lines = o3d.geometry.LineSet()
            for room_node in self.room_nodes:
                room_positions = room_node.mask
                room_node_pos = np.mean(room_positions, axis=0)
                room_node_pos[2] = self.floor_height + 1.8

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(room_node_pos)
                room_spheres += sphere

                number_lable = create_number_point_cloud(room_node.room_id, room_node_pos)
                room_labels += number_lable

                for node in room_node.nodes:
                    node_pos = node.position
                    node_pos[2] = self.floor_height
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([room_node_pos, node_pos])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
                    room_viewpoint_lines += line

            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}', exist_ok=True)
            o3d.io.write_triangle_mesh(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/room_spheres.ply', room_spheres)
            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/room_labels.ply', room_labels)
            o3d.io.write_line_set(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/room_viewpoint_lines.ply', room_viewpoint_lines)

            whole_scene_pcd = self.navigable_pcd + self.useful_pcd
            whole_scene_pcd_points = whole_scene_pcd.point.positions.cpu().numpy()
            whole_scene = o3d.geometry.PointCloud()
            whole_scene.points = o3d.utility.Vector3dVector(whole_scene_pcd_points)
            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/whole_scene.ply', whole_scene)

            o3d.io.write_triangle_mesh(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/centers.ply', combined_sphere)

            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/centers_label.ply', combined_labels)

            # # save the frontier info
            # node_info = {}
            # for node in self.nodes:
            #     node_info[node.idx] = {
            #         'idx': node.idx,
            #         'position': node.position.tolist(),
            #         'has_frontier': node.has_frontier,
            #         'has_true_frontier': node.has_true_frontier,
            #         'frontier_idxs': (node.frontier_idxs.tolist()),
            #     }
            # node_info = json.dumps(node_info)
            # with open(f'{self.save_dir}/episode-{episode_idx}/whole/step_{step_idx}/info.txt', 'w') as f:
            #     f.write(f'{node_info}')
        except Exception as e:
            logger.info('Failed to save whole visualization: {}', e)

        try:
            # save node state
            info = self.to_json_save_node_info()
            # save the info in a json file and put nice parse:
            os.makedirs(f'{self.save_dir}/episode-{episode_idx}/node_info', exist_ok=True)
            with open(f'{self.save_dir}/episode-{episode_idx}/node_info/node_info_{step_idx}.json', 'w') as f:
                json.dump(info, f, indent=4)
        except Exception as e:
            logger.info('Failed to save node state: {}', e)

    def get_nodes(self, temporary_pcd, angles, node, episode_idx, step):
        # only keep the max connected component in traversable area
        self.traversable_pcd = self.keep_the_max_connect_component(self.traversable_pcd_all)

        current_node_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        for i in range(len(temporary_pcd)):
            current_node_pcd = gpu_merge_pointcloud(current_node_pcd, temporary_pcd[i])

        # # save current_node_pcd
        # save_pcd = o3d.geometry.PointCloud()
        # save_pcd.points = o3d.utility.Vector3dVector(current_node_pcd.point.positions.cpu().numpy())
        # save_pcd.colors = o3d.utility.Vector3dVector(current_node_pcd.point.colors.cpu().numpy())
        # os.makedirs(f'{self.save_dir}/episode-{episode_idx}/current_node_pcd', exist_ok=True)
        # o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/current_node_pcd/current_node_pcd_{step}.ply', save_pcd)

        current_node_pcd_position = current_node_pcd.point.positions.cpu().numpy()
        angles_current_node = np.arctan2(current_node_pcd_position[:, 1] - self.current_position[1],
                                         current_node_pcd_position[:, 0] - self.current_position[0])
        angles_current_node = np.where(angles_current_node < 0, angles_current_node + 2 * np.pi,
                                       angles_current_node)
        angles_current_node = np.where(angles_current_node > 2 * np.pi,
                                       angles_current_node - 2 * np.pi, angles_current_node)

        if node is not None:
            pcd_stored_in_node = node.pcd
            try:
                if pcd_stored_in_node.point.positions.shape[0] > 0:
                    nav_pcd_stored_in_node = pcd_stored_in_node.select_by_index(
                        (pcd_stored_in_node.point.positions[:, 2]
                         < self.ceiling_height).nonzero()[0])
                    nav_pcd_stored_in_node = nav_pcd_stored_in_node.select_by_index(
                        (nav_pcd_stored_in_node.point.positions[:, 2]
                         < self.floor_height + 0.2).nonzero()[0])

                    nav_pcd_stored_in_node_position = nav_pcd_stored_in_node.point.positions.cpu(
                    ).numpy()
                    nav_pcd_stored_in_node_position[:, 2] = np.ones_like(
                        nav_pcd_stored_in_node_position[:, 2]) * np.mean(
                            nav_pcd_stored_in_node_position[:, 2])
                    nav_pcd_stored_in_node.point["positions"] = o3d.core.Tensor(
                        nav_pcd_stored_in_node_position,
                        dtype=o3d.core.Dtype.Float32,
                        device=self.pcd_device)

                    self.process_nav_pcd = gpu_merge_pointcloud(self.process_nav_pcd,
                                                                nav_pcd_stored_in_node)

                process_nav_pcd_position = self.process_nav_pcd.point.positions.cpu().numpy()
                process_nav_pcd_position[:, 2] = np.ones_like(
                    process_nav_pcd_position[:, 2]) * np.mean(process_nav_pcd_position[:, 2])
                self.process_nav_pcd.point["positions"] = o3d.core.Tensor(
                    process_nav_pcd_position, dtype=o3d.core.Dtype.Float32, device=self.pcd_device)
                self.process_nav_pcd = self.process_nav_pcd.voxel_down_sample(self.pcd_resolution)

                if pcd_stored_in_node.point.positions.shape[0] > 0:
                    obstcale_pcd_stored_in_node = pcd_stored_in_node.select_by_index(
                        (pcd_stored_in_node.point.positions[:, 2]
                         > self.floor_height + 0.2).nonzero()[0])
                    self.process_obs_pcd = gpu_merge_pointcloud(
                        self.process_obs_pcd,
                        obstcale_pcd_stored_in_node).voxel_down_sample(self.pcd_resolution)
            except:
                pcd_stored_in_node = None
        else:
            pcd_stored_in_node = None

        floor_height = self.floor_height

        # obstcale_pcd_points = self.obstacle_pcd.point.positions.cpu().numpy()

        obstacle_pcd = self.process_obs_pcd.select_by_index(
            (self.process_obs_pcd.point.positions[:, 2] > self.floor_height + 0.2).nonzero()[0])
        obstacle_pcd = obstacle_pcd.select_by_index((obstacle_pcd.point.positions[:, 2]
                                                     < self.ceiling_height).nonzero()[0])

        obstacle_pcd_points = obstacle_pcd.point.positions.cpu().numpy()

        current_navigable_pcd = self.process_nav_pcd

        # # save current_node_pcd
        # save_pcd = o3d.geometry.PointCloud()
        # save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
        # save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
        # os.makedirs(f'{self.save_dir}/episode-{episode_idx}/current_node_pcd', exist_ok=True)
        # o3d.io.write_point_cloud(f'{self.save_dir}/episode-{episode_idx}/current_node_pcd/current_node_pcd_{step}.ply', save_pcd)

        current_position = self.current_position

        current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()
        all_points_num = current_navigable_position.shape[0]

        # change the z value of the points to the mean of the floor points
        obstacle_pcd_points[:, 2] = np.ones_like(obstacle_pcd_points[:, 2]) * np.mean(
            current_navigable_position[:, 2])

        standing_position = np.array(
            [current_position[0], current_position[1], current_navigable_position[:, 2].mean()])

        closest_distance = 1.35

        # get the frontier_clusters
        # self-implemented function
        # frontier_clusters, frontier_centers = self.get_frontiers(current_navigable_pcd, current_pcd.point.positions.cpu().numpy())

        # offered function
        # frontier_clusters, frontier_centers = self.get_frontiers_offerd(
        #     obstacle_pcd, current_navigable_pcd, self.floor_height + 0.2, 0.1,
        #     closest_distance * 0.5)
        global_frontier_clusters, global_frontier_centers, global_frontier_map_idxs, frontier_map_idxs_all = self.get_frontiers_offerd(
            self.obstacle_pcd, self.navigable_pcd, self.floor_height + 0.2, 0.1,
            closest_distance*0.7)
        frontier_clusters = global_frontier_clusters
        frontier_centers = global_frontier_centers

        self.current_global_frontier_map_idxs = frontier_map_idxs_all
        # print(global_frontier_map_idxs)

        # save the frontier clusters
        if len(frontier_clusters) != 0:
            frontiers_to_save = np.concatenate(frontier_clusters, axis=0)
        else:
            frontiers_to_save = np.array([]).reshape(0, 3)

        logger.info("Current position: {}", standing_position)
        # logger.info(len(current_navigable_position))

        # decide the max distance
        if len(frontier_clusters) != 0:
            frontiers_all = np.concatenate(frontier_clusters, axis=0)
            distance_to_frontiers = np.linalg.norm(frontiers_all - standing_position, axis=1)
            min_distance_to_frontiers = np.min(distance_to_frontiers)
            logger.info(f"Min distance to frontiers: {min_distance_to_frontiers}")
            max_distance = min(2.5, min_distance_to_frontiers * 1.2)
        else:
            max_distance = 2.5

        logger.info(f"Max distance Threshold: {max_distance}")
        intersections = self.calculate_intersections(current_navigable_position,
                                                     standing_position,
                                                     max_distance=max_distance)
        # extand the dimention of the intersections to 3
        intersections = np.concatenate((intersections, np.ones(
            (intersections.shape[0], 1)) * np.mean(current_navigable_position[:, 2])),
                                       axis=1)

        polygon_path = Path(intersections[:, :2])
        # Remove points inside the polygon
        points = current_navigable_position[:, :2]
        mask = 1 - polygon_path.contains_points(points)

        # Ensure the mask is a tensor on the correct device
        mask_tensor = o3d.core.Tensor(np.where(mask)[0],
                                      o3d.core.Dtype.Int64,
                                      device=current_navigable_pcd.device)

        # Select points by index
        pcd_removed = current_navigable_pcd.select_by_index(mask_tensor)

        mask_tensor_in_circle = o3d.core.Tensor(np.where(1 - mask)[0],
                                                o3d.core.Dtype.Int64,
                                                device=current_navigable_pcd.device)
        pcd_in_circle = current_navigable_pcd.select_by_index(mask_tensor_in_circle)
        colors = np.random.rand(3)
        pcd_in_circle.paint_uniform_color(colors)

        # cluster the remaing points
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd_removed.cluster_dbscan(eps=0.08, min_points=4, print_progress=False))

        # logger.info(labels)
        if step != 12 and len(labels) == 0:
            return
        max_label = labels.max().cpu().numpy()

        # extract each cluster and rule out the small ones caluclate the center of each cluster
        clusters = []
        centers_from_frontiers = []
        centers_from_frontiers_map_idxs = []
        centers_from_clusters = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
                                              o3d.core.Dtype.Int64,
                                              device=current_navigable_pcd.device)
            cluster = pcd_removed.select_by_index(mask_idx_tensor)
            cluster_point = cluster.point.positions.cpu().numpy()
            # generate random colors, shape is (n, 3)
            colors = np.random.rand(3)
            cluster.paint_uniform_color(colors)

            # extract frontiers in this cluster
            current_frontier_clusters = []
            current_frontier_centers = []
            current_frontier_map_idxs = []
            for frontier_index, frontier_cluster in enumerate(frontier_clusters):
                distances = np.linalg.norm(frontier_cluster[:, np.newaxis, :2] -
                                           cluster_point[np.newaxis, :, :2],
                                           axis=2)
                if np.min(distances) < 0.2:
                    # frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.15]
                    # frontier_center_new = np.mean(frontier_cluster_new, axis=0)
                    frontier_cluster_new = frontier_cluster
                    frontier_center_new = frontier_centers[frontier_index]
                    frontier_cluster_map_idxs_new = global_frontier_map_idxs[frontier_index]
                    current_frontier_clusters.append(frontier_cluster_new)
                    current_frontier_centers.append(frontier_center_new)
                    current_frontier_map_idxs.append(frontier_cluster_map_idxs_new)

            if len(current_frontier_clusters) != 0 and len(cluster_point) > 50:
                clusters.append(cluster)
                # merge the frontier clusters
                merged_frontier_clusters, merged_frontier_centers, merged_frontier_map_idxs, line_set = self.merge_frontier_with_visibility_1(
                    cluster_point,
                    obstacle_pcd_points,
                    current_frontier_clusters,
                    current_frontier_centers,
                    current_frontier_map_idxs,
                    current_position,
                )

                os.makedirs(f'{self.save_dir}/episode-{episode_idx}/frontier_line', exist_ok=True)
                o3d.io.write_line_set(
                    f'{self.save_dir}/episode-{episode_idx}/frontier_line/line_set_{step}_cluster_{i}.ply',
                    line_set)

                center_idx_to_remove = set()  # 改用 set 避免重复
                for index1 in range(len(merged_frontier_centers)):
                    for index2 in range(index1 + 1, len(merged_frontier_centers)):
                        if index1 in center_idx_to_remove or index2 in center_idx_to_remove:
                            continue  # 跳过已标记删除的索引

                        center1 = merged_frontier_centers[index1]
                        center2 = merged_frontier_centers[index2]
                        angle1 = np.arctan2(center1[1] - standing_position[1],
                                            center1[0] - standing_position[0])
                        angle2 = np.arctan2(center2[1] - standing_position[1],
                                            center2[0] - standing_position[0])
                        angle_diff = np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi)

                        if angle_diff < 10 * np.pi / 180 and self.is_visible(
                                center1, center2, cluster_point, obstacle_pcd_points):
                            distance1 = np.linalg.norm(center1[:2] - standing_position[:2])
                            distance2 = np.linalg.norm(center2[:2] - standing_position[:2])

                            if distance1 < distance2:
                                center_idx_to_remove.add(index2)
                            else:
                                center_idx_to_remove.add(index1)

                # 从大到小排序，避免索引偏移问题
                center_idx_to_remove = sorted(center_idx_to_remove, reverse=True)
                centers_new = [center for idx, center in enumerate(merged_frontier_centers) if
                               idx not in center_idx_to_remove]
                centers_map_idxs_new = [map_idx for idx, map_idx in enumerate(merged_frontier_map_idxs) if
                                        idx not in center_idx_to_remove]

                centers_from_frontiers.extend(centers_new)  # 直接使用 extend 替代 for-loop
                centers_from_frontiers_map_idxs.extend(centers_map_idxs_new)

            # cluster without frontiers
            if len(current_frontier_clusters) == 0 and (
                (all_points_num / 20 < len(cluster.point.positions.cpu().numpy())) or
                (50 < len(cluster.point.positions.cpu().numpy()))):
                # determine whether there are frontiers in the cluster
                flag = True
                nav_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
                cluster_points = cluster.point.positions.cpu().numpy()
                nav_grid_idxs = translate_point_to_grid(cluster_points, self.grid_resolution, self.voxel_dimension)
                nav_map[nav_grid_idxs[:, 0], nav_grid_idxs[:, 1]] = 1
                node_positions = np.array([node.position for node in self.nodes])
                node_grid_idxs = translate_point_to_grid(node_positions, self.grid_resolution, self.voxel_dimension)
                for node_grid_idx in node_grid_idxs:
                    if nav_map[node_grid_idx[0], node_grid_idx[1]] == 1:
                        flag = False
                        break

                if flag:
                    clusters.append(cluster)
                    center = np.mean(cluster.point.positions.cpu().numpy(), axis=0)
                    centers_from_clusters.append(center)

        current_nav_pcd_to_save = o3d.t.geometry.PointCloud(self.pcd_device)
        current_nav_pcd_to_save = gpu_merge_pointcloud(current_nav_pcd_to_save, pcd_in_circle, False)
        for cluster in clusters:
            current_nav_pcd_to_save = gpu_merge_pointcloud(current_nav_pcd_to_save, cluster, False)

        # compute the distance between the current centers and self.nodes,
        # if the distance is less than a threshold, then remove the center
        # and add the new center to self.nodes
        centers_from_frontiers = np.array(centers_from_frontiers)
        centers_from_clusters = np.array(centers_from_clusters)
        valid_centers = []

        # check if fail at the beginning
        if len(centers_from_frontiers) + len(centers_from_clusters) == 0:
            logger.info(f'No centers found in step {step}')

        nodes_positions = self.get_nodes_positions()
        # logger.info(f'Nodes positions: {nodes_positions}')
        for idx, center in enumerate(centers_from_frontiers):
            frontier_idxs = centers_from_frontiers_map_idxs[idx]
            # ensure the center is in the traversable area, if not use the closest point in the traversable area
            center = self.find_closest_point_in_pc(center, self.traversable_pcd)
            if center is None: continue
            traversable_flag = self.check_traversability(current_position, center)

            if traversable_flag:
                # distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
                # # logger.info(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
                # if np.min(distance) > closest_distance * 0.7:

                center[2] = self.current_position[2]
                valid_centers.append((center, True, frontier_idxs))

        for center in centers_from_clusters:
            # ensure the center is in the traversable area, if not use the closest point in the traversable area
            center = self.find_closest_point_in_pc(center, self.traversable_pcd)
            if center is None: continue
            traversable_flag = self.check_traversability(current_position, center)

            if traversable_flag:
                # distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
                # # logger.info(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
                # if np.min(distance) > closest_distance * 0.7:

                center[2] = self.current_position[2]
                valid_centers.append((center, False, np.array([])))

        if len(valid_centers) > 0:
            block_current = (len(valid_centers) == 1)
            for center, has_frontier, frontier_idxs in valid_centers:
                angle_center = np.arctan2(center[1] - self.current_position[1],
                                          center[0] - self.current_position[0])
                angle_center = np.where(angle_center < 0, angle_center + 2 * np.pi, angle_center)
                angle_center = np.where(angle_center > 2 * np.pi, angle_center - 2 * np.pi,
                                        angle_center)

                start_angle = self.normalize_angle(angle_center - 40 * np.pi / 180)
                end_angle = self.normalize_angle(angle_center + 40 * np.pi / 180)
                if start_angle < end_angle:
                    mask_idx_tensor = o3d.core.Tensor(
                        np.where((angles_current_node > start_angle) &
                                 (angles_current_node < end_angle))[0],
                        o3d.core.Dtype.Int64,
                        device=self.pcd_device)
                    pcd_to_node = current_node_pcd.select_by_index(mask_idx_tensor)
                else:
                    mask_idx_tensor = o3d.core.Tensor(
                        np.where((angles_current_node > start_angle) |
                                 (angles_current_node < end_angle))[0],
                        o3d.core.Dtype.Int64,
                        device=self.pcd_device)
                    pcd_to_node = current_node_pcd.select_by_index(mask_idx_tensor)

                _, add_node_flag = self.add_node(center, pcd_to_node, has_frontier, frontier_idxs, block_current)
                if add_node_flag:
                    self.add_edge(self.current_node_idx, self.nodes[-1].idx)

                    # for index in self.current_obj_indices:
                    #     nwobj = self.objects[index]
                    #     objpos = nwobj.position
                    #     objpos[2] = self.floor_height
                    #
                    #     obj_angle = np.arctan2(objpos[1] - self.current_position[1],
                    #                            objpos[0] - self.current_position[0])
                    #     obj_angle = np.where(obj_angle < 0, obj_angle + 2 * np.pi, obj_angle)
                    #     obj_angle = np.where(obj_angle > 2 * np.pi, obj_angle - 2 * np.pi, obj_angle)
                    #
                    #     is_visible = self.is_visible_obj(center, objpos, self.obstacle_pcd.point.positions.cpu().numpy())
                    #
                    #     if (np.abs(obj_angle - angle_center) < 30 * np.pi / 180 or np.abs(obj_angle - angle_center) > 330 * np.pi / 180) and is_visible:
                    #         self.update_obj(center, [index])

        # update the obj, node edges
        self.update_obj_node()

        valid_centers = np.array([center for center, _, _ in valid_centers])

        # check the visibility of the nodes in the graph
        self.update_edges()

        self.segment_room(step)

        self.update_node_true_frontier()
        self.update_room_state()

        # save the point cloud
        self.save_point_cloud(pcd_stored_in_node, self.process_obs_pcd, frontiers_to_save,
                              intersections, current_nav_pcd_to_save, self.traversable_pcd,
                              valid_centers, episode_idx, step)

    def update_obj_node(self):
        # update the obj, node edges
        tmp_pcd = self.useful_pcd.select_by_index(
            (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.88).nonzero()[0])

        for node in self.nodes:
            center = node.position
            node_idx = node.idx
            node_obj_idxs = []
            for index, obj in enumerate(self.objects):
                nwobj = self.objects[index]
                objpos = nwobj.position
                objpos[2] = self.floor_height

                is_visible = self.is_visible(center, objpos, self.navigable_pcd.point.positions.cpu().numpy(), tmp_pcd.point.positions.cpu().numpy())

                if is_visible:
                    node_obj_idxs.append(index)
            self.update_obj(node_idx, node_obj_idxs)


    # def get_nodes_process(self, node, idx, path_idx, step):
    #     # only keep the max connected component in traversable area
    #     self.traversable_pcd = self.keep_the_max_connect_component(self.traversable_pcd)
    #
    #     current_node_pcd = self.process_obs_pcd.select_by_index(
    #         (self.process_obs_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])
    #     current_node_pcd_position = current_node_pcd.point.positions.cpu().numpy()
    #     angles_current_node = np.arctan2(current_node_pcd_position[:, 1] - self.current_position[1],
    #                                      current_node_pcd_position[:, 0] - self.current_position[0])
    #     angles_current_node = np.where(angles_current_node < 0, angles_current_node + 2 * np.pi,
    #                                    angles_current_node)
    #     angles_current_node = np.where(angles_current_node > 2 * np.pi,
    #                                    angles_current_node - 2 * np.pi, angles_current_node)
    #
    #     floor_height = self.floor_height
    #
    #     # obstcale_pcd_points = self.obstacle_pcd.point.positions.cpu().numpy()
    #     try:
    #         obstacle_pcd = self.process_obs_pcd.select_by_index(
    #             (self.process_obs_pcd.point.positions[:, 2] > self.floor_height + 0.2).nonzero()[0])
    #         obstacle_pcd = obstacle_pcd.select_by_index((obstacle_pcd.point.positions[:, 2]
    #                                                      < self.ceiling_height).nonzero()[0])
    #         obstacle_pcd_points = obstacle_pcd.point.positions.cpu().numpy()
    #     except:
    #         obstacle_pcd_points = np.array([])
    #
    #     current_navigable_pcd = self.process_nav_pcd
    #     current_position = self.current_position
    #
    #     current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()
    #     all_points_num = current_navigable_position.shape[0]
    #
    #     save_pcd = o3d.geometry.PointCloud()
    #     save_pcd.points = o3d.utility.Vector3dVector(
    #         current_navigable_pcd.point.positions.cpu().numpy())
    #     save_pcd.colors = o3d.utility.Vector3dVector(
    #         current_navigable_pcd.point.colors.cpu().numpy())
    #     import os
    #     os.makedirs(f'{self.save_dir}/episode-{idx}/nav', exist_ok=True)
    #     o3d.io.write_point_cloud(f'{self.save_dir}/episode-{idx}/nav/nav_{step}_{path_idx}.ply',
    #                              save_pcd)
    #
    #     # change the z value of the points to the mean of the floor points
    #     obstacle_pcd_points[:, 2] = np.ones_like(obstacle_pcd_points[:, 2]) * np.mean(
    #         current_navigable_position[:, 2])
    #
    #     standing_position = np.array(
    #         [current_position[0], current_position[1], current_navigable_position[:, 2].mean()])
    #
    #     closest_distance = 1.35
    #
    #     # frontier_clusters, frontier_centers = self.get_frontiers_offerd(
    #     #     obstacle_pcd, current_navigable_pcd, self.floor_height + 0.2, 0.1,
    #     #     closest_distance * 0.67)
    #
    #     global_frontier_clusters, global_frontier_centers, global_frontier_map_idxs = self.get_frontiers_offerd(
    #         self.obstacle_pcd, self.navigable_pcd, self.floor_height + 0.2, 0.1,
    #         closest_distance*0.7)
    #     frontier_clusters = global_frontier_clusters
    #     frontier_centers = global_frontier_centers
    #
    #     # frontier_clusters_whole, frontier_centers_whole = self.get_frontiers_offerd(obstacle_pcd,
    #     #                                                                             current_navigable_pcd,
    #     #                                                                             self.floor_height + 0.2,
    #     #                                                                             0.1, closest_distance)
    #     #
    #     # # extract frontiers in current observation
    #     # frontier_clusters = []
    #     # frontier_centers = []
    #     # for frontier_index, frontier_cluster in enumerate(frontier_clusters_whole):
    #     #     distances = np.linalg.norm(
    #     #         frontier_cluster[:, np.newaxis, :2] - current_navigable_position[np.newaxis, :, :2],
    #     #         axis=2)
    #     #     if np.min(distances) < 0.2:
    #     #         frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.2]
    #     #         frontier_center_new = np.mean(frontier_cluster_new, axis=0)
    #     #         frontier_clusters.append(frontier_cluster_new)
    #     #         frontier_centers.append(frontier_center_new)
    #
    #     if len(frontier_clusters) != 0:
    #         # logger.info('Frontier clusters', (frontier_clusters))
    #         frontiers_to_save = np.concatenate(frontier_clusters, axis=0)
    #         save_pcd = o3d.geometry.PointCloud()
    #         save_pcd.points = o3d.utility.Vector3dVector(frontiers_to_save)
    #         save_pcd.paint_uniform_color([0, 1, 0])
    #         import os
    #         os.makedirs(f'{self.save_dir}/episode-{idx}/frontier', exist_ok=True)
    #         o3d.io.write_point_cloud(
    #             f'{self.save_dir}/episode-{idx}/frontier/frontier_{step}_{path_idx}.ply', save_pcd)
    #     else:
    #         return
    #
    #     # decide the max distance
    #     if len(frontier_clusters) != 0:
    #         frontiers_all = np.concatenate(frontier_clusters, axis=0)
    #         distance_to_frontiers = np.linalg.norm(frontiers_all - standing_position, axis=1)
    #         min_distance_to_frontiers = np.min(distance_to_frontiers)
    #         logger.info(f"Min distance to frontiers: {min_distance_to_frontiers}")
    #         max_distance = min(2.5, min_distance_to_frontiers * 1.2)
    #
    #     intersections = self.calculate_intersections(current_navigable_position,
    #                                                  standing_position,
    #                                                  max_distance=max_distance)
    #
    #     # extand the dimention of the intersections to 3
    #     intersections = np.concatenate((intersections, np.ones(
    #         (intersections.shape[0], 1)) * np.mean(current_navigable_position[:, 2])),
    #                                    axis=1)
    #
    #     polygon_path = Path(intersections[:, :2])
    #     # Remove points inside the polygon
    #     points = current_navigable_position[:, :2]
    #     mask = 1 - polygon_path.contains_points(points)
    #
    #     # Ensure the mask is a tensor on the correct device
    #     mask_tensor = o3d.core.Tensor(np.where(mask)[0],
    #                                   o3d.core.Dtype.Int64,
    #                                   device=current_navigable_pcd.device)
    #
    #     # Select points by index
    #     pcd_removed = current_navigable_pcd.select_by_index(mask_tensor)
    #
    #     mask_tensor_in_circle = o3d.core.Tensor(np.where(1 - mask)[0],
    #                                             o3d.core.Dtype.Int64,
    #                                             device=current_navigable_pcd.device)
    #     pcd_in_circle = current_navigable_pcd.select_by_index(mask_tensor_in_circle)
    #     colors = np.random.rand(3)
    #     pcd_in_circle.paint_uniform_color(colors)
    #
    #     # cluster the remaining points
    #     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #         labels = np.array(
    #             pcd_removed.cluster_dbscan(eps=0.1, min_points=4, print_progress=False))
    #
    #     # logger.info(labels)
    #     if step != 12 and len(labels) == 0:
    #         return
    #     max_label = labels.max().cpu().numpy()
    #     # logger.info(f"point cloud has {max_label + 1} clusters")
    #
    #     # extract each cluster and rule out the small ones caluclate the center of each cluster
    #     clusters = []
    #     centers_from_frontiers = []
    #     centers_from_frontiers_map_idxs = []
    #     centers_from_clusters = []
    #     for i in range(max_label + 1):
    #         mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
    #                                           o3d.core.Dtype.Int64,
    #                                           device=current_navigable_pcd.device)
    #         cluster = pcd_removed.select_by_index(mask_idx_tensor)
    #         cluster_point = cluster.point.positions.cpu().numpy()
    #         # generate random colors, shape is (n, 3)
    #         colors = np.random.rand(3)
    #         cluster.paint_uniform_color(colors)
    #
    #         # extract frontiers in this cluster
    #         current_frontier_clusters = []
    #         current_frontier_centers = []
    #         current_frontier_map_idxs = []
    #         for frontier_index, frontier_cluster in enumerate(frontier_clusters):
    #             distances = np.linalg.norm(frontier_cluster[:, np.newaxis, :2] -
    #                                        cluster_point[np.newaxis, :, :2],
    #                                        axis=2)
    #             if np.min(distances) < 0.2:
    #                 # frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.15]
    #                 # frontier_center_new = np.mean(frontier_cluster_new, axis=0)
    #                 frontier_cluster_new = frontier_cluster
    #                 frontier_center_new = frontier_centers[frontier_index]
    #                 frontier_cluster_map_idxs_new = global_frontier_map_idxs[frontier_index]
    #                 current_frontier_clusters.append(frontier_cluster_new)
    #                 current_frontier_centers.append(frontier_center_new)
    #                 current_frontier_map_idxs.append(frontier_cluster_map_idxs_new)
    #
    #         if len(current_frontier_clusters) != 0 and len(cluster_point) > 50:
    #             clusters.append(cluster)
    #             # merge the frontier clusters
    #             merged_frontier_clusters, merged_frontier_centers, merged_frontier_map_idxs, non_merged_frontier_map_idxs, line_set = self.merge_frontier_with_visibility_1(
    #                 cluster_point,
    #                 obstacle_pcd_points,
    #                 current_frontier_clusters,
    #                 current_frontier_centers,
    #                 current_frontier_map_idxs,
    #                 current_position,
    #             )
    #
    #             os.makedirs(f'{self.save_dir}/episode-{idx}/frontier_line', exist_ok=True)
    #             o3d.io.write_line_set(
    #                 f'{self.save_dir}/episode-{idx}/frontier_line/line_set_{step}_{path_idx}_cluster_{i}.ply',
    #                 line_set)
    #
    #             center_idx_to_remove = set()  # 改用 set 避免重复
    #             for index1 in range(len(merged_frontier_centers)):
    #                 for index2 in range(index1 + 1, len(merged_frontier_centers)):
    #                     if index1 in center_idx_to_remove or index2 in center_idx_to_remove:
    #                         continue  # 跳过已标记删除的索引
    #
    #                     center1 = merged_frontier_centers[index1]
    #                     center2 = merged_frontier_centers[index2]
    #                     angle1 = np.arctan2(center1[1] - standing_position[1],
    #                                         center1[0] - standing_position[0])
    #                     angle2 = np.arctan2(center2[1] - standing_position[1],
    #                                         center2[0] - standing_position[0])
    #                     angle_diff = np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi)
    #
    #                     if angle_diff < 10 * np.pi / 180 and self.is_visible(
    #                             center1, center2, cluster_point, obstacle_pcd_points):
    #                         distance1 = np.linalg.norm(center1[:2] - standing_position[:2])
    #                         distance2 = np.linalg.norm(center2[:2] - standing_position[:2])
    #
    #                         if distance1 < distance2:
    #                             center_idx_to_remove.add(index2)
    #                         else:
    #                             center_idx_to_remove.add(index1)
    #
    #             # 从大到小排序，避免索引偏移问题
    #             center_idx_to_remove = sorted(center_idx_to_remove, reverse=True)
    #             centers_new = [center for idx, center in enumerate(merged_frontier_centers) if
    #                            idx not in center_idx_to_remove]
    #             centers_map_idxs_new = [map_idx for idx, map_idx in enumerate(merged_frontier_map_idxs) if
    #                                     idx not in center_idx_to_remove]
    #
    #             centers_from_frontiers.extend(centers_new)  # 直接使用 extend 替代 for-loop
    #             centers_from_frontiers_map_idxs.extend(centers_map_idxs_new)
    #         # if len(current_frontier_clusters) == 0 and (
    #         #         (all_points_num / 20 < len(cluster.point.positions.cpu().numpy()) < all_points_num / 5) or (
    #         #         50 < len(cluster.point.positions.cpu().numpy()) < 200)):
    #         #     clusters.append(cluster)
    #         #     center = np.mean(cluster.point.positions.cpu().numpy(), axis=0)
    #         #     centers_from_clusters.append(center)
    #
    #     current_pcd_to_save = o3d.t.geometry.PointCloud(self.pcd_device)
    #     current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, pcd_in_circle)
    #     for cluster in clusters:
    #         current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, cluster)
    #     save_pcd = o3d.geometry.PointCloud()
    #     save_pcd.points = o3d.utility.Vector3dVector(
    #         current_pcd_to_save.point.positions.cpu().numpy())
    #     save_pcd.colors = o3d.utility.Vector3dVector(current_pcd_to_save.point.colors.cpu().numpy())
    #     import os
    #     os.makedirs(f'{self.save_dir}/episode-{idx}/nav', exist_ok=True)
    #     o3d.io.write_point_cloud(f'{self.save_dir}/episode-{idx}/nav/nav_{step}_{path_idx}.ply',
    #                              save_pcd)
    #
    #     centers_from_frontiers = np.array(centers_from_frontiers)
    #     centers_from_clusters = np.array(centers_from_clusters)
    #     valid_centers = []
    #
    #     nodes_positions = self.get_nodes_positions()
    #     # logger.info(f'Nodes positions: {nodes_positions}')
    #     for idx, center in enumerate(centers_from_frontiers):
    #         frontier_idxs = centers_from_frontiers_map_idxs[idx]
    #         # ensure the center is in the traversable area, if not use the closest point in the traversable area
    #         center = self.find_closest_point_in_pc(center, self.traversable_pcd)
    #         if center is None: continue
    #         traversable_flag = self.check_traversability(current_position, center)
    #
    #         if traversable_flag:
    #             # distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
    #             # # logger.info(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
    #             # if np.min(distance) > closest_distance * 0.7:
    #
    #             center[2] = self.current_position[2]
    #             valid_centers.append((center, True, frontier_idxs))
    #
    #     for center in centers_from_clusters:
    #         # ensure the center is in the traversable area, if not use the closest point in the traversable area
    #         center = self.find_closest_point_in_pc(center, self.traversable_pcd)
    #         if center is None: continue
    #         traversable_flag = self.check_traversability(current_position, center)
    #
    #         if traversable_flag:
    #             # distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
    #             # # logger.info(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
    #             # if np.min(distance) > closest_distance * 0.7:
    #
    #             center[2] = self.current_position[2]
    #             valid_centers.append((center, False, np.array([])))
    #
    #     # logger.info(self.nodes)
    #     # logger.info(valid_centers)
    #     if len(valid_centers) > 0:
    #         block_current = (len(valid_centers) == 1)
    #         for center, has_frontier, frontier_idxs in valid_centers:
    #             angle_center = np.arctan2(center[1] - self.current_position[1],
    #                                       center[0] - self.current_position[0])
    #             angle_center = np.where(angle_center < 0, angle_center + 2 * np.pi, angle_center)
    #             angle_center = np.where(angle_center > 2 * np.pi, angle_center - 2 * np.pi,
    #                                     angle_center)
    #             # # find the closet index in the angles
    #             # angle_diff = np.abs(angles - angle_center)
    #             # index = np.argmin(angle_diff)
    #             # pcd_to_node = temporary_pcd[index]
    #
    #             start_angle = self.normalize_angle(angle_center - 40 * np.pi / 180)
    #             end_angle = self.normalize_angle(angle_center + 40 * np.pi / 180)
    #             if start_angle < end_angle:
    #                 mask_idx_tensor = o3d.core.Tensor(
    #                     np.where((angles_current_node > start_angle) &
    #                              (angles_current_node < end_angle))[0],
    #                     o3d.core.Dtype.Int64,
    #                     device=self.pcd_device)
    #                 pcd_to_node = current_node_pcd.select_by_index(mask_idx_tensor)
    #             else:
    #                 mask_idx_tensor = o3d.core.Tensor(
    #                     np.where((angles_current_node > start_angle) |
    #                              (angles_current_node < end_angle))[0],
    #                     o3d.core.Dtype.Int64,
    #                     device=self.pcd_device)
    #                 pcd_to_node = current_node_pcd.select_by_index(mask_idx_tensor)
    #
    #             _, add_node_flag = self.add_node(center, pcd_to_node, has_frontier, frontier_idxs, block_current)
    #             if add_node_flag:
    #                 self.add_edge(self.current_position, center)
    #
    #     # remove the processed pcd
    #     self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #     self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #
    #     valid_centers = np.array([center for center, _, _ in valid_centers])
    #
    #     # check the visibility of the nodes in the graph
    #     if step != 12:
    #         self.update_edges(self.navigable_pcd, self.current_position, valid_centers)
    #
    #     self.update_node_frontier()
    #     self.segment_room(step)

    def get_frontiers_offerd(self,
                             obstacle_pcd,
                             navigable_pcd,
                             obstacle_height=-0.7,
                             grid_resolution=0.1,
                             closest_distance=1.6):



        grid_map_new = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))

        frontier_map, frontiers = project_frontier(obstacle_pcd, navigable_pcd, obstacle_height, self.grid_resolution, self.voxel_dimension)

        frontiers[:, 2] = np.ones_like(frontiers[:, 2]) * np.mean(
            np.array(navigable_pcd.point.positions.cpu().numpy())[:, 2])

        current_position = self.current_position[:2]
        distance_frontiers = np.linalg.norm(frontiers[:, :2] - np.array(current_position), axis=1)
        frontiers = frontiers[distance_frontiers > closest_distance]

        frontiers_pcd = o3d.t.geometry.PointCloud()
        frontiers_pcd.point.positions = o3d.core.Tensor(frontiers, device=frontiers_pcd.device)

        # cluster the frontiers
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                frontiers_pcd.cluster_dbscan(eps=0.3, min_points=3, print_progress=False))

        if len(labels) == 0:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(
                    frontiers_pcd.cluster_dbscan(eps=0.3, min_points=1, print_progress=False))
            if len(labels) == 0:
                return [], [], [], []
        max_label = labels.max().cpu().numpy()
        # extract each cluster and rule out the small ones caluclate the center of each cluster
        frontier_clusters = []
        frontier_centers = []
        frontier_map_idxs = []
        frontier_map_idxs_all = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
                                              o3d.core.Dtype.Int64,
                                              device=frontiers_pcd.device)
            frontier_cluster = frontiers_pcd.select_by_index(mask_idx_tensor)
            frontier_cluster_points = frontier_cluster.point.positions.cpu().numpy()
            frontier_cluster_map_idxs = translate_point_to_grid(frontier_cluster_points, self.grid_resolution, self.voxel_dimension)[:, :2]

            grid_map_new[frontier_cluster_map_idxs[:, 0], frontier_cluster_map_idxs[:, 1]] = 1
            # if 80% of the points are already marked 1 in the grid map, then ignore this cluster
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print(frontier_cluster_map_idxs.shape)
            # print(np.sum(self.frontiers_considered[frontier_cluster_map_idxs[:, 0], frontier_cluster_map_idxs[:, 1]]),
            #       0.8 * len(frontier_cluster_map_idxs))
            frontier_map_idxs_all.append(frontier_cluster_map_idxs)
            if np.sum(self.frontiers_considered[frontier_cluster_map_idxs[:, 0], frontier_cluster_map_idxs[:, 1]]) > 0.8 * len(frontier_cluster_map_idxs):
                continue

            frontier_clusters.append(frontier_cluster.point.positions.cpu().numpy())
            frontier_center = np.mean(frontier_cluster.point.positions.cpu().numpy(), axis=0)
            frontier_centers.append(frontier_center)
            frontier_map_idxs.append(frontier_cluster_map_idxs)

        self.grid_map = grid_map_new

        self.update_node_frontier()

        logger.info(f'Total number of clusters: {len(frontier_clusters)}')

        return frontier_clusters, frontier_centers, frontier_map_idxs, frontier_map_idxs_all

    def is_visible(self, point1, point2, cluster_points, obstacle_points):
        point1 = point1.copy()
        point2 = point2.copy()
        cluster_points = cluster_points.copy()
        obstacle_points = obstacle_points.copy()

        nav_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        obstacle_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))

        # find the closet point in cluster_points to point1 and point2
        distance1 = np.linalg.norm(cluster_points[:, :2] - point1[:2], axis=1)
        distance2 = np.linalg.norm(cluster_points[:, :2] - point2[:2], axis=1)

        point1 = cluster_points[np.argmin(distance1)]
        point2 = cluster_points[np.argmin(distance2)]

        point1_map = translate_point_to_grid(point1, self.grid_resolution, self.voxel_dimension)[:2]
        point2_map = translate_point_to_grid(point2, self.grid_resolution, self.voxel_dimension)[:2]
        cluster_points_map = translate_point_to_grid(cluster_points, self.grid_resolution, self.voxel_dimension)[:, :2]
        obstacle_points_map = translate_point_to_grid(obstacle_points, self.grid_resolution, self.voxel_dimension)[:, :2]

        nav_map[cluster_points_map[:, 0], cluster_points_map[:, 1]] = 1
        obstacle_map[obstacle_points_map[:, 0], obstacle_points_map[:, 1]] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        obstacle_map = cv2.dilate(obstacle_map, kernel, iterations=1)
        obstacle_map = cv2.bitwise_not(obstacle_map)
        nav_map = cv2.bitwise_and(nav_map, obstacle_map)

        interpolate_points = bresenham(point1_map[0], point1_map[1], point2_map[0], point2_map[1])
        interpolate_points = list(interpolate_points)[1:-1]

        for point in interpolate_points:
            if nav_map[point[0], point[1]] == 0:
                return False

        return True

    def is_visible_in_nav_map(self, point1, point2, cluster_points):
        point1_map = translate_point_to_grid(point1, self.grid_resolution, self.voxel_dimension)[:2]
        point2_map = translate_point_to_grid(point2, self.grid_resolution, self.voxel_dimension)[:2]
        cluster_points_map = translate_point_to_grid(cluster_points, self.grid_resolution, self.voxel_dimension)[:, :2]

        nav_map = np.zeros((self.voxel_dimension[0], self.voxel_dimension[1]))
        nav_map[cluster_points_map[:, 0], cluster_points_map[:, 1]] = 1

        interpolate_points = bresenham(point1_map[0], point1_map[1], point2_map[0], point2_map[1])

        for point in interpolate_points:
            if nav_map[point[0], point[1]] == 0:
                return False

        return True

    def is_out_of_boundary_frontier_cluster(self,
                                            point_cloud,
                                            obstacle_points,
                                            points,
                                            radius=0.1,
                                            density_threshold=1):
        # 这里假设有一个函数，检查点是否在点云的边界外
        # 例如可以使用点云的凸包或距离来判断
        radius = radius
        density_threshold = density_threshold
        # 创建KD树以加速邻域搜索
        tree = KDTree(point_cloud)
        obstcale_tree = KDTree(obstacle_points)

        for point in points:
            # 查询半径内的点
            indices = tree.query_ball_point(point, radius)
            obstacle_indices = obstcale_tree.query_ball_point(point, radius / 2)
            # 统计邻域内点的数量
            local_density = len(indices) - 1
            obstacle_density = len(obstacle_indices) - 1
            if local_density < density_threshold or obstacle_density > 0:
                return True

        return False

    def merge_frontier_with_visibility_1(self,
                                         cluster_points,
                                         obstacle_points,
                                         frontier_clusters,
                                         frontier_centers,
                                         frontier_map_idxs,
                                         current_position=None):

        obstacle_points[:, 2] = np.ones_like(obstacle_points[:, 2]) * np.mean(cluster_points[:, 2])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frontier_centers)
        lines = []

        G = nx.Graph()
        for i in range(len(frontier_centers)):
            G.add_node(i)
            for j in range(i + 1, len(frontier_centers)):
                # check visibility
                if self.is_visible(frontier_centers[i], frontier_centers[j], cluster_points,
                                   obstacle_points):
                    G.add_edge(i, j)
                    lines.append([i, j])

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0])

        cliques = list(nx.find_cliques(G))

        # logger.info("All cliques in the graph:", cliques)

        # 合并一个cluster中的frontier_cluster
        merged_frontier_clusters = []
        merged_frontier_centers = []
        merged_frontier_map_idxs = []
        for cluster in cliques:
            merged_cluster = np.empty((0, 3))
            merged_cluster_center = np.empty((0, 3))
            merged_cluster_map_idx = np.empty((0, 2)).astype(int)
            # if len(cluster) > 1:
            for idx in cluster:
                # logger.info(frontier_clusters[idx])
                merged_cluster = np.concatenate((merged_cluster, frontier_clusters[idx]), axis=0)
                merged_cluster_center = np.concatenate(
                    (merged_cluster_center, np.array([frontier_centers[idx]])), axis=0)
                merged_cluster_map_idx = np.concatenate(
                    (merged_cluster_map_idx, frontier_map_idxs[idx]), axis=0)

            merged_frontier_clusters.append(merged_cluster)
            merged_frontier_centers.append(np.mean(merged_cluster_center, axis=0))
            merged_frontier_map_idxs.append(merged_cluster_map_idx)
            # else:
            #     merged_cluster = np.concatenate((merged_cluster, frontier_clusters[idx]), axis=0)
            #     distance

        return merged_frontier_clusters, merged_frontier_centers, merged_frontier_map_idxs, line_set

    def change_state(self, node):
        nodes_states = self.get_nodes_states()
        logger.info(f'Node state before: {nodes_states}')

        self.visit_node(node.idx)

        nodes_states = self.get_nodes_states()
        logger.info(f'Node state after: {nodes_states}')
        # node_idx = np.where((self.nodes == node).all(axis=1))[0]
        # logger.info(f'Node index: {node_idx}')
        # logger.info(f'Node state before: {self.nodes_state}')
        # logger.info(f'Selected nodes {self.nodes[node_idx]}')
        # self.nodes_state[node_idx] = 1
        # logger.info(f'Node state after: {self.nodes_state}')

    # def get_candidate_node(self, instruct_goal, idx=None, step=None):
    #     self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #     self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #
    #     current_node = self.find_closest_node(self.current_position)
    #     current_node_idx = current_node.idx
    #     current_node_pos = current_node.position
    #
    #     current_state_info = f"You are now at node {current_node_idx} with position {current_node_pos}."
    #
    #     traj_info = 'The robot history trajectory is: '
    #     for traj_idx in self.traj:
    #         traj_info += f'Viewpoint {traj_idx} Position {self.nodes[traj_idx].position} --> '
    #     traj_info = traj_info[:-4]
    #
    #     json_dict = self.to_json()
    #     #transfer this dict to string
    #     json_str = json.dumps(json_dict)
    #
    #     prompt_info = instruct_goal + '\n' + current_state_info + '\n' + traj_info + '\n' + json_str
    #
    #     os.makedirs(f'{self.save_dir}/episode-{idx}/gpt', exist_ok=True)
    #     with open(f'{self.save_dir}/episode-{idx}/gpt/prompt_{step}.txt', 'w') as f:
    #         f.write(f'Prompt_info: {prompt_info}\n')
    #         f.write(f'\n')
    #
    #     client = OpenAI()
    #     class Step(BaseModel):
    #         explanation: str
    #         output: str
    #
    #     class Reasoning(BaseModel):
    #         steps: list[Step]
    #         final_answer: int
    #         final_object: int
    #         flag: bool
    #
    #     completion = client.beta.chat.completions.parse(
    #         model="gpt-4o-2024-08-06",
    #         messages=[
    #             {"role": "system",
    #              "content": CHAINON_PROMPT},
    #             {"role": "user", "content": prompt_info}
    #         ],
    #         response_format=Reasoning,
    #     )
    #
    #     answer = completion.choices[0].message.parsed
    #     # save the answer to a txt file
    #     with open(f'{self.save_dir}/episode-{idx}/gpt/answer_{step}.txt', 'w') as f:
    #         f.write(f'Answer: {answer}\n')
    #         f.write(f'\n')
    #         f.write(f'\n')
    #     with open(f'{self.save_dir}/episode-{idx}/gpt/input_{step}.txt', 'a') as f:
    #         f.write(f'Input: {instruct_goal}\n' + json_str)
    #         f.write(f'\n')
    #         f.write(f'\n')
    #
    #     logger.info(f'Answer: {answer}')
    #     print(f'Answer: {answer}')
    #
    #     waypoint_final = self.nodes[answer.final_answer]
    #     try:
    #         assert waypoint_final.state == 0
    #     except:
    #         with open(f'{self.save_dir}/episode-{idx}/gpt_failure.txt', 'w') as f:
    #             f.write(f'GPT failed to find an unexplored node\n')
    #
    #     # ratings = answer.final_answer
    #     # nodes_states = self.get_nodes_states()
    #     # # choose the node with the highest rating and node state is 0
    #     # max_rating = -1
    #     # max_rating_idx = -1
    #     # for idx, rating in enumerate(ratings):
    #     #     if rating > max_rating and nodes_states[idx] == 0:
    #     #         max_rating = rating
    #     #         max_rating_idx = idx
    #     #
    #     # if max_rating_idx == -1:
    #     #     return None, None, False
    #     #
    #     # waypoint_final = self.nodes[max_rating_idx]
    #
    #     if answer.final_object != -1 and answer.flag:
    #         object_final = self.objects[answer.final_object]
    #     else:
    #         object_final = None
    #
    #     return waypoint_final, object_final, answer.flag

    def get_candidate_room_fully_explored(self, instruct_goal, idx=None, step=None):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        current_node = self.nodes[self.current_node_idx]
        current_node_idx = current_node.idx
        current_node_pos = current_node.position

        room_idx = current_node.room_idx

        current_state_info = f"You are now at node with position {current_node_pos} in the Room {room_idx}."

        traj_info = 'The robot history trajectory is: '
        for traj_idx in self.traj:
            traj_info += f'Position {self.nodes[traj_idx].position} --> '
        traj_info = traj_info[:-4]

        json_dict = self.to_json()
        #transfer this dict to string
        json_str = json.dumps(json_dict)

        candidate_room_idxs = [room_node.room_id for room_node in self.room_nodes if room_node.state == 0]
        choose_instruction = f'Please choose a room to explore from the following rooms: {candidate_room_idxs}'

        prompt_info = instruct_goal + '\n' + current_state_info + '\n' + traj_info + '\n' + json_str + '\n' + choose_instruction

        os.makedirs(f'{self.save_dir}/episode-{idx}/gpt_room', exist_ok=True)
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/prompt_{step}.txt', 'w') as f:
            # logger.info(f'Prompt info: {prompt_info}')

            f.write(f'Prompt_info: {prompt_info}\n')
            f.write(f'\n')

        class Step(BaseModel):
            explanation: str
            output: str

        class Reasoning(BaseModel):
            steps: list[Step]
            final_answer: int
            reason: str

        if self.vlm == 'gemini':
            try:
                client = OpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

                completion = client.beta.chat.completions.parse(
                    # model="gemini-2.5-pro-exp-03-25",
                    model=MODEL_NAME,
                    messages=[{
                        "role": "system",
                        "content": ROOM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt_info
                    }],
                    response_format=Reasoning,
                )
            except Exception as e:
                logger.error(f'Gemini limit: {e}')
                client = OpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

                completion = client.beta.chat.completions.parse(
                    model="gemini-2.0-flash",
                    messages=[{
                        "role": "system",
                        "content": ROOM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt_info
                    }],
                    response_format=Reasoning,
                )
        if self.vlm == 'openai':
            client = OpenAI()

            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{
                    "role": "system",
                    "content": ROOM_PROMPT
                }, {
                    "role": "user",
                    "content": prompt_info
                }],
                response_format=Reasoning,
            )

        answer = completion.choices[0].message.parsed
        # save the answer to a txt file
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/answer_{step}.txt', 'w') as f:
            f.write(f'Answer: {answer}\n')
            f.write(f'\n')
            f.write(f'\n')
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/input_{step}.txt', 'w') as f:
            f.write(f'Input: {instruct_goal}\n' + json_str)
            f.write(f'\n')
            f.write(f'\n')

        logger.info(f'Answer: {answer}')

        room_final = self.room_nodes[answer.final_answer]

        try:
            assert room_final.state == 0
        except:
            with open(f'{self.save_dir}/episode-{idx}/gpt_failure.txt', 'w') as f:
                f.write(f'GPT failed to find an unexplored room\n')

        return room_final

    def get_candidate_room_fully_explored_no_gpt(self, instruct_goal, idx=None, step=None):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        current_node = self.nodes[self.current_node_idx]
        current_node_idx = current_node.idx
        current_node_pos = current_node.position

        room_idx = current_node.room_idx

        current_state_info = f"You are now at node with position {current_node_pos} in the Room {room_idx}."

        traj_info = 'The robot history trajectory is: '
        for traj_idx in self.traj:
            traj_info += f'Position {self.nodes[traj_idx].position} --> '
        traj_info = traj_info[:-4]

        json_dict = self.to_json()
        #transfer this dict to string
        json_str = json.dumps(json_dict)

        candidate_room_idxs = [room_node.room_id for room_node in self.room_nodes if room_node.state == 0]
        choose_instruction = f'Please choose a room to explore from the following rooms: {candidate_room_idxs}'

        prompt_info = instruct_goal + '\n' + current_state_info + '\n' + traj_info + '\n' + json_str + '\n' + choose_instruction

        os.makedirs(f'{self.save_dir}/episode-{idx}/gpt_room', exist_ok=True)
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/prompt_{step}.txt', 'w') as f:
            # logger.info(f'Prompt info: {prompt_info}')

            f.write(f'Prompt_info: {prompt_info}\n')
            f.write(f'\n')

        frontier_nodes_pos = [node.position for node in self.nodes if node.has_frontier == 1]
        frontier_nodes_idx = [node.idx for node in self.nodes if node.has_frontier == 1]
        # calculate the distance between the current position and the frontier nodes using self.env.sim.geodesic_distance
        nodes_positions = np.array(frontier_nodes_pos)
        nodes_positions = nodes_positions + self.initial_position
        nodes_positions[:, 2] = self.initial_position[2] - 0.88
        nodes_positions = nodes_positions[:, [0, 2, 1]]

        current_node_position = self.current_position + self.initial_position
        current_node_position = np.array([
            current_node_position[0], self.initial_position[2] - 0.88, current_node_position[1]
        ])
        distance = [
            self.env.sim.geodesic_distance(current_node_position, node_position)
            for node_position in nodes_positions
        ]

        closest_node = self.nodes[frontier_nodes_idx[np.argmin(distance)]]
        room_idx = closest_node.room_idx
        room_final = self.room_nodes[room_idx]

        answer = f"""
                Distance to {frontier_nodes_idx} are {distance}.
                Node {closest_node.idx} in Room {room_idx} is the closest node to the current position.
                """

        logger.info(f'Answer: {answer}')

        with open(f'{self.save_dir}/episode-{idx}/gpt_room/answer_{step}.txt', 'w') as f:
            f.write(f'Answer: {answer}\n')
            f.write(f'\n')
            f.write(f'\n')

        try:
            assert room_final.state == 0
        except:
            with open(f'{self.save_dir}/episode-{idx}/gpt_failure.txt', 'w') as f:
                f.write(f'GPT failed to find an unexplored room\n')

        return room_final

    def object_found(self, instruct_goal, idx=None, step=None):
        json_dict = self.to_json_wo_some_class()
        #transfer this dict to string
        json_str = json.dumps(json_dict)

        prompt_info = instruct_goal + '\n' + json_str

        os.makedirs(f'{self.save_dir}/episode-{idx}/gpt_obj', exist_ok=True)
        with open(f'{self.save_dir}/episode-{idx}/gpt_obj/prompt_{step}.txt', 'w') as f:
            f.write(f'Prompt_info: {prompt_info}\n')
            f.write(f'\n')

        class Step(BaseModel):
            explanation: str
            output: str

        class Reasoning(BaseModel):
            steps: list[Step]
            flag: bool
            final_object: int

        if self.vlm == 'gemini':
            client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

            completion = client.beta.chat.completions.parse(
                model="gemini-2.0-flash",
                messages=[{
                    "role": "system",
                    "content": OBJECT_PROMPT
                }, {
                    "role": "user",
                    "content": prompt_info
                }],
                response_format=Reasoning,
            )
        if self.vlm == 'openai':
            client = OpenAI()

            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{
                    "role": "system",
                    "content": OBJECT_PROMPT
                }, {
                    "role": "user",
                    "content": prompt_info
                }],
                response_format=Reasoning,
            )

        answer = completion.choices[0].message.parsed
        # save the answer to a txt file
        with open(f'{self.save_dir}/episode-{idx}/gpt_obj/answer_{step}.txt', 'w') as f:
            f.write(f'Answer: {answer}\n')
            f.write(f'\n')
            f.write(f'\n')
        with open(f'{self.save_dir}/episode-{idx}/gpt_obj/input_{step}.txt', 'a') as f:
            f.write(f'Input: {instruct_goal}\n' + json_str)
            f.write(f'\n')
            f.write(f'\n')

        logger.info(f'Answer: {answer}')

        if not answer.flag:
            obj_final = None
        else:
            obj_final = self.objects[answer.final_object]
            rgb = obj_final.rgb
            bbox = obj_final.bbox

        return answer.flag, obj_final

    def object_found_no_gpt(self, instruct_goal, idx=None, step=None):
        json_dict = self.to_json_wo_some_class()
        #transfer this dict to string
        json_str = json.dumps(json_dict)
        prompt_info = instruct_goal + '\n' + json_str

        os.makedirs(f'{self.save_dir}/episode-{idx}/no_gpt_obj', exist_ok=True)
        with open(f'{self.save_dir}/episode-{idx}/no_gpt_obj/prompt_{step}.txt', 'w') as f:
            f.write(f'Prompt_info: {prompt_info}\n')
            f.write(f'\n')
        with open(f'{self.save_dir}/episode-{idx}/no_gpt_obj/input_{step}.txt', 'a') as f:
            f.write(f'Input: {prompt_info}\n')
            f.write(f'\n')
            f.write(f'\n')

        target_objs = []
        for obj in self.objects:
            tag = obj.tag
            if tag == self.target:
                target_objs.append(obj)
        if len(target_objs) == 0:
            answer = f"No target object '{self.target}' found"
            with open(f'{self.save_dir}/episode-{idx}/no_gpt_obj/answer_{step}.txt', 'w') as f:
                f.write(f'Input: {prompt_info}\n')
                f.write(f'Answer: {answer}\n')
                f.write(f'\n')
                f.write(f'\n')
            return False, None
        # choose the object with the highest confidence
        target_objs = sorted(target_objs, key=lambda x: x.confidence.numpy().item(), reverse=True)
        target_obj = target_objs[0]

        answer = f"Choose Obj '{target_obj.tag}' at position {target_obj.position} with confidence {target_obj.confidence.numpy().item()}."
        with open(f'{self.save_dir}/episode-{idx}/no_gpt_obj/answer_{step}.txt', 'w') as f:
            f.write(f'Input: {prompt_info}\n')
            f.write(f'Answer: {answer}\n')
            f.write(f'\n')
            f.write(f'\n')

        return True, target_obj

    def get_candidate_room_relocate(self, instruct_goal, idx=None, step=None):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        current_node = self.nodes[self.current_node_idx]
        current_node_idx = current_node.idx
        current_node_pos = current_node.position

        room_idx = current_node.room_idx

        current_state_info = f"You are now at node with position {current_node_pos} in the Room {room_idx}."

        traj_info = 'The robot history trajectory is: '
        for traj_idx in self.traj:
            traj_info += f'Position {self.nodes[traj_idx].position} --> '
        traj_info = traj_info[:-4]

        json_dict = self.to_json()
        #transfer this dict to string
        json_str = json.dumps(json_dict)

        prompt_info = instruct_goal + '\n' + current_state_info + '\n' + traj_info + '\n' + json_str

        os.makedirs(f'{self.save_dir}/episode-{idx}/gpt_room', exist_ok=True)
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/prompt_{step}.txt', 'w') as f:
            # logger.info(f'Prompt info: {prompt_info}')

            f.write(f'Prompt_info: {prompt_info}\n')
            f.write(f'\n')

        class Step(BaseModel):
            explanation: str
            output: str

        class Reasoning(BaseModel):
            steps: list[Step]
            flag: bool
            final_answer: int
            reason: str

        if self.vlm == 'gemini':
            client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

            completion = client.beta.chat.completions.parse(
                model="gemini-2.0-flash",
                messages=[{
                    "role": "system",
                    "content": RELOCATE_PROMPT
                }, {
                    "role": "user",
                    "content": prompt_info
                }],
                response_format=Reasoning,
            )
        if self.vlm == 'openai':
            client = OpenAI()

            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{
                    "role": "system",
                    "content": RELOCATE_PROMPT
                }, {
                    "role": "user",
                    "content": prompt_info
                }],
                response_format=Reasoning,
            )

        answer = completion.choices[0].message.parsed
        # save the answer to a txt file
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/answer_{step}.txt', 'w') as f:
            f.write(f'Answer: {answer}\n')
            f.write(f'\n')
            f.write(f'\n')
        with open(f'{self.save_dir}/episode-{idx}/gpt_room/input_{step}.txt', 'w') as f:
            f.write(f'Input: {instruct_goal}\n' + json_str)
            f.write(f'\n')
            f.write(f'\n')

        logger.info(f'Answer: {answer}')

        relocate_flag = answer.flag
        if relocate_flag:
            room_final = self.room_nodes[answer.final_answer]
            try:
                assert room_final.state == 0
            except:
                with open(f'{self.save_dir}/episode-{idx}/gpt_failure.txt', 'w') as f:
                    f.write(f'GPT failed to find an unexplored room\n')
        else:
            room_final = None

        return relocate_flag, room_final

    def get_candidate_node_nearest(self, tmp, idx=None, step=None):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        return self.find_closest_unexplored_node(), -1, False

    def get_path(self, end):

        return np.array([end.position]), [end.idx]

        # path_node_position, path_node_idx = self.find_the_closest_path(self.current_position, end.position)
        # # calculate the geometric distance along the path
        # path_distance = 0
        # for i in range(len(path_node_position) - 1):
        #     pos1 = path_node_position[i]
        #     pos2 = path_node_position[i + 1]
        #     pos1 = pos1 + self.initial_position
        #     pos2 = pos2 + self.initial_position
        #     pos1[2] = self.initial_position[2]-0.88
        #     pos2[2] = self.initial_position[2]-0.88
        #     pos1 = [pos1[0], pos1[2], pos1[1]]
        #     pos2 = [pos2[0], pos2[2], pos2[1]]
        #     path_distance += self.env.sim.geodesic_distance(pos1, pos2)
        # # calculate the geometric distance from the current position to the end position
        # current_position = self.current_position
        # current_position = current_position + self.initial_position
        # current_position[2] = self.initial_position[2]-0.88
        # current_position = [current_position[0], current_position[2], current_position[1]]
        # end_position = end.position + self.initial_position
        # end_position[2] = self.initial_position[2]-0.88
        # end_position = [end_position[0], end_position[2], end_position[1]]
        # distance_to_end = self.env.sim.geodesic_distance(current_position, end_position)
        #
        # if path_distance > distance_to_end * 2:
        #     # discard the path
        #     return np.array([end.position]), [end.idx]
        # else:
        #     return path_node_position, path_node_idx

    # def choose_waypoint(self, candidate_nodes):
    #     node_idx = np.argmin(
    #         [np.linalg.norm(np.array([node[0], node[1]]) - self.current_position[:2]) for node in
    #          candidate_nodes])
    #     node = candidate_nodes[node_idx]
    #
    #     self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #
    #     return node

    def get_closest_disances_and_points(self,
                                        current_pcd_position,
                                        whole_pcd_position,
                                        current_position,
                                        max_distance=1.7):
        distance_current = np.linalg.norm(current_pcd_position - current_position, axis=1)
        distance_whole = np.linalg.norm(whole_pcd_position - current_position, axis=1)

        # # avoid noise
        # pcd_position = pcd_position[distance > 0.3]
        # distance = distance[distance > 0.3]
        #
        # # avoid noise
        # if len(pcd_position[distance < 0.55]) < 10:
        #     pcd_position = pcd_position[distance > 0.55]
        #     distance = distance[distance > 0.55]

        # logger.info(f'Min distance: {np.min(distance)}')
        # logger.info(f'Max distance: {np.max(distance)}')

        angles_current = np.arctan2(current_pcd_position[:, 1] - current_position[1],
                                    current_pcd_position[:, 0] - current_position[0])
        angles_current = np.where(angles_current < 0, angles_current + 2 * np.pi, angles_current)
        angles_current = np.where(angles_current > 2 * np.pi, angles_current - 2 * np.pi,
                                  angles_current)

        angles_whole = np.arctan2(whole_pcd_position[:, 1] - current_position[1],
                                  whole_pcd_position[:, 0] - current_position[0])
        angles_whole = np.where(angles_whole < 0, angles_whole + 2 * np.pi, angles_whole)
        angles_whole = np.where(angles_whole > 2 * np.pi, angles_whole - 2 * np.pi, angles_whole)

        start_angle = np.min(angles_current)
        end_angle = np.max(angles_current)

        angle_resolution = (2 * np.pi) / 360
        if end_angle - start_angle < 2 * np.pi - 0.1:
            angle_bins = np.arange(np.min(angles_current),
                                   np.max(angles_current) - 3 * angle_resolution, angle_resolution)
        else:
            angles_1 = angles_current[angles_current < np.pi]
            angles_2 = angles_current[angles_current > np.pi]
            start_angle_1 = np.min(angles_1)
            end_angle_1 = np.max(angles_1)
            start_angle_2 = np.min(angles_2)
            end_angle_2 = np.max(angles_2)
            angle_bins_1 = np.arange(start_angle_1, end_angle_1 - 3 * angle_resolution,
                                     angle_resolution)
            angle_bins_2 = np.arange(start_angle_2, end_angle_2, angle_resolution)
            angle_bins = np.concatenate((angle_bins_1, angle_bins_2), axis=0)

        # logger.info(f'Angle bins: {angle_bins/ (2 * np.pi) * 360}')
        nearest_distances = np.full_like(angle_bins, max_distance, dtype=float)
        nearest_points = np.empty((0, 3))

        for i, angle in enumerate(angle_bins):
            pcd_position_within_range = current_pcd_position[(angles_current > angle) & (
                angles_current < (angle + 3 * angle_resolution) % (2 * np.pi))]
            distance_within_range = distance_current[(angles_current > angle) & (
                angles_current < (angle + 3 * angle_resolution) % (2 * np.pi))]

            whole_pcd_position_within_range = whole_pcd_position[(angles_whole > angle) & (
                angles_whole < (angle + 3 * angle_resolution) % (2 * np.pi))]
            distance_whole_within_range = distance_whole[(angles_whole > angle) & (
                angles_whole < (angle + 3 * angle_resolution) % (2 * np.pi))]

            if len(pcd_position_within_range) != 0:
                if np.min(distance_within_range) < nearest_distances[i]:
                    current_distance_min = np.min(distance_within_range)
                    if len(whole_pcd_position_within_range) == 0:
                        whole_distance_min = 100
                    else:
                        whole_distance_min = np.min(distance_whole_within_range)

                    if current_distance_min < whole_distance_min:
                        nearest_distances[i] = np.min(distance_within_range)
                        nearest_point = pcd_position_within_range[distance_within_range < (
                            np.min(distance_within_range) + 0.06)]
                    else:
                        nearest_distances[i] = np.min(distance_whole_within_range)
                        nearest_point = whole_pcd_position_within_range[
                            distance_whole_within_range < (np.min(distance_whole_within_range) +
                                                           0.06)]

                    # # select the k nearest points in pcd_position_within_range
                    # nearest_point_k_nearest = pcd_position_within_range[np.argsort(distance_within_range)[:2]]
                    # # merge nearest_point_avoid_noise and nearest_point and remove the duplicate points
                    # nearest_point = np.concatenate((nearest_point, nearest_point_k_nearest), axis=0)
                    # nearest_point = np.unique(nearest_point, axis=0)

                    if len(nearest_point) > 0:
                        nearest_points = np.concatenate((nearest_points, nearest_point), axis=0)

        # save_pcd = o3d.geometry.PointCloud()
        # save_pcd.points = o3d.utility.Vector3dVector(nearest_points)
        # save_pcd.paint_uniform_color([1, 0, 0])
        # import os
        # os.makedirs(f'{self.save_dir}/episode-0/inter', exist_ok=True)
        # o3d.io.write_point_cloud(f'{self.save_dir}/episode-{0}/inter/inter_{self.update_iterations}.ply', save_pcd)

        return nearest_distances, nearest_points

    def normalize_angle(self, angle):
        if angle < 0:
            angle += 2 * np.pi
        if angle > 2 * np.pi:
            angle -= 2 * np.pi
        return angle

    # def to_json(self):
    #     json_data = {}
    #     object_json = []
    #     for i in range(len(self.objects)):
    #         obj = self.objects[i]
    #         nwpcd = obj.pcd.point.positions.cpu().numpy()
    #         size = np.zeros(3)
    #         size[0] = np.max(nwpcd[:, 0]) - np.min(nwpcd[:, 0])
    #         size[1] = np.max(nwpcd[:, 1]) - np.min(nwpcd[:, 1])
    #         size[2] = np.max(nwpcd[:, 2]) - np.min(nwpcd[:, 2])
    #
    #         object_json.append({
    #             'index': i,
    #             'position': obj.position.tolist(),
    #             'class': obj.tag,
    #             'confidence': obj.confidence.numpy().item(),
    #             'size': size.tolist(),
    #         })
    #     json_data['objects'] = object_json
    #
    #     nodes = []
    #     for i in range(self.node_cnt):
    #         node = self.nodes[i]
    #         if node.state == NodeState.EXPLORED:
    #             node.has_frontier = False
    #         nodes.append({
    #             'idx': i,
    #             'position': node.position.tolist(),
    #             'state': node.state,
    #             'neighbors': self.neighbors[i],
    #             'has_frontier': node.has_frontier,
    #         })
    #
    #         # nwobj = []
    #         # for obj_idx in node.objects:
    #         #     obj = self.objects[obj_idx]
    #         #     nwobj.append({
    #         #         'idx': obj_idx,
    #         #         'position': obj.position.tolist(),
    #         #         'class': obj.tag,
    #         #         'confidence': obj.confidence.numpy().item(),
    #         #     })
    #
    #         nodes[-1]['objects'] = node.objects
    #
    #     json_data['nodes'] = nodes
    #
    #     return json_data

    def to_json(self):
        json_data = {}
        object_json = []
        for i in range(len(self.objects)):
            obj = self.objects[i]
            nwpcd = obj.pcd.point.positions.cpu().numpy()
            size = np.zeros(3)
            size[0] = np.max(nwpcd[:, 0]) - np.min(nwpcd[:, 0])
            size[1] = np.max(nwpcd[:, 1]) - np.min(nwpcd[:, 1])
            size[2] = np.max(nwpcd[:, 2]) - np.min(nwpcd[:, 2])

            object_json.append({
                'index': i,
                'position': [round(p, 3) for p in obj.position.tolist()],
                'class': obj.tag,
                'confidence': round(obj.confidence.numpy().item(), 3),
                'size': [round(s, 3) for s in size.tolist()],
            })
        json_data['objects'] = object_json

        room_nodes = []
        for i in range(len(self.room_nodes)):
            room_node = self.room_nodes[i]
            nodes_in_room = room_node.nodes
            nodes = []
            for j in range(len(nodes_in_room)):
                node = nodes_in_room[j]
                if node.state == NodeState.EXPLORED:
                    node.has_frontier = False
                    node.has_true_frontier = False
                nodes.append({
                    # 'idx': node.idx,
                    'position': [round(p, 3) for p in node.position.tolist()],
                    # 'state': node.state,
                    # 'neighbors': self.neighbors[node.idx],
                    'has_frontier': node.has_frontier,
                })

                nodes[-1]['objects'] = node.objects
            room_nodes.append({
                'room_idx': room_node.room_id,
                'state': room_node.state,
                'distance': round(room_node.distance, 3),
                'viewpoints': nodes,
            })

        json_data['Room'] = room_nodes

        return json_data

    def to_json_wo_some_class(self):
        json_data = {}
        object_json = []
        for i in range(len(self.objects)):
            obj = self.objects[i]
            nwpcd = obj.pcd.point.positions.cpu().numpy()
            size = np.zeros(3)
            size[0] = np.max(nwpcd[:, 0]) - np.min(nwpcd[:, 0])
            size[1] = np.max(nwpcd[:, 1]) - np.min(nwpcd[:, 1])
            size[2] = np.max(nwpcd[:, 2]) - np.min(nwpcd[:, 2])

            if obj.tag != 'unknown' and obj.tag != 'furniture':
                object_json.append({
                    'index': i,
                    'position': obj.position.tolist(),
                    'class': obj.tag,
                    'confidence': obj.confidence.numpy().item(),
                    'size': size.tolist(),
                })
        json_data['objects'] = object_json

        return json_data

    def find_closest_point_in_pc(self, position, pcd):
        pcd_positions = pcd.point.positions.cpu().numpy()
        distance = np.linalg.norm(pcd_positions[:, :2] - position[:2], axis=1)
        min_distance = np.min(distance)
        min_distance_idx = np.argmin(distance)
        chosen_position = pcd_positions[min_distance_idx]

        # find the closest point in self.navigable_pcd with position
        navigable_pcd_pos = self.navigable_pcd.point.positions.cpu().numpy()
        distance = np.linalg.norm(navigable_pcd_pos[:, :2] - position[:2], axis=1)
        min_distance = np.min(distance)
        min_distance_idx = np.argmin(distance)
        position_in_nav = navigable_pcd_pos[min_distance_idx]

        is_visible_flag = self.is_visible_in_nav_map(chosen_position, position_in_nav, navigable_pcd_pos)
        if min_distance > 1.0 or not is_visible_flag:
            logger.info(f'Abort this viewpoint, {chosen_position}')
            return None
        else:
            return chosen_position

    # TODO: may need some modification
    def keep_the_max_connect_component(self, pcd):
        if self.episode_idx == 55:
            # resolution = 0.05
            # dimension = self.voxel_dimension * 2
            resolution = 0.1
            dimension = self.voxel_dimension

            trav_map = np.zeros((dimension[0], dimension[1]))
            pcd_points = pcd.point.positions.cpu().numpy()
            avg_z = np.mean(pcd_points[:, 2])
            pcd_idxs = translate_point_to_grid(pcd_points, resolution, dimension)[:, :2]
            trav_map[pcd_idxs[:, 0], pcd_idxs[:, 1]] = 1

            # find the max connect component
            trav_map = np.array(trav_map, dtype=np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(trav_map, connectivity=8)
            # save trav_map as a image
            # cv2.imwrite('trav_map.png', trav_map*255)
            if num_labels == 2:
                return pcd
            max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            # get the x, y coordinates of the max connect component
            mask = np.zeros(trav_map.shape, dtype=np.uint8)
            mask[labels == max_label] = 1
            map_idxs = np.argwhere(mask == 1)
            # expand the map_idxs to the 3d space
            map_idxs = np.concatenate([map_idxs, np.ones((map_idxs.shape[0], 1))*avg_z], axis=1)
            pcd_points = translate_grid_to_point(map_idxs, resolution, dimension)
            pcd_colors = np.ones_like(pcd_points) * 100

            pcd_new = gpu_pointcloud_from_array(pcd_points, pcd_colors, self.pcd_device)

            return pcd_new



        # use the dbscan to cluster the points
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.12, min_points=12, print_progress=False))
        if len(labels) == 0:
            return pcd
        max_label = labels.max().cpu().numpy()
        # extract the max connect component
        max_cluster = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0],
                                              o3d.core.Dtype.Int64,
                                              device=pcd.device)
            cluster = pcd.select_by_index(mask_idx_tensor)
            max_cluster.append(cluster)
        max_cluster = max_cluster[np.argmax(
            [len(cluster.point.positions.cpu().numpy()) for cluster in max_cluster])]
        return max_cluster

    def check_traversability(self, start, end):
        start[2] = self.floor_height
        end[2] = self.floor_height

        start = start + self.initial_position
        start = [start[0], start[2], start[1]]
        end = end + self.initial_position
        end = [end[0], end[2], end[1]]

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end

        found_path = self.env.sim.pathfinder.find_path(path)

        if found_path:
            return True
        else:
            return False


    def segment_room(self, step_idx):
        self.room_nodes = []
        for node in self.nodes:
            node.room_idx = -1

        obs_pcd = self.scene_pcd.select_by_index((self.scene_pcd.point.positions[:, 2]
                                                  < self.ceiling_height).nonzero()[0])
        nav_pcd = self.navigable_pcd
        tmp_floor_path = f'{self.save_dir}/episode-{self.episode_idx}/room_inter/step_{self.update_iterations}'
        os.makedirs(tmp_floor_path, exist_ok=True)

        save_intermediate_results = True

        obs_pcd = obs_pcd.voxel_down_sample(voxel_size=0.05)
        nav_pcd = nav_pcd.voxel_down_sample(voxel_size=0.05)

        # floor_pcd.voxel_down_sample(voxel_size=0.05)
        xyz = obs_pcd.point.positions.cpu().numpy()
        nav_points = nav_pcd.point.positions.cpu().numpy()
        xyz = np.concatenate([xyz, nav_points], axis=0)

        # print(xyz)
        xyz_full = xyz.copy()
        floor_zero_level = -0.8
        floor_height = 0.8
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 2] < floor_height]
        xyz = xyz[xyz[:, 2] >= floor_zero_level + 0.2]
        xyz_full = xyz_full[xyz_full[:, 2] < floor_height]

        # project the point cloud to 2d
        # pcd_2d = xyz[:, [0, 1]]
        # xyz_full = xyz_full[:, [0, 1]]
        pcd_2d = xyz
        xyz_full = xyz_full

        room_resolution = 0.05
        # room_dimension = self.voxel_dimension
        # room_dimension[0] *= (self.grid_resolution // room_resolution)
        # room_dimension[1] *= (self.grid_resolution // room_resolution)

        room_dimension = np.asarray([1000,1000,20])

        hist = project_room(pcd_2d,grid_resolution=room_resolution,voxel_dimension=room_dimension)
        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "obstacale_2D_histogram.png"), hist)

        # applythresholding
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton_hist = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # apply closing to the walls skeleton
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton_hist = cv2.morphologyEx(walls_skeleton_hist,
                                               cv2.MORPH_CLOSE,
                                               kernal,
                                               iterations=1)

        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "walls_skeleton.png"), walls_skeleton_hist)

        # extract outside boundary from histogram of xyz_full
        hist_full = project_room(xyz_full,grid_resolution=room_resolution,voxel_dimension=room_dimension)

        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, outside_boundary = cv2.threshold(hist_full, 0.001, 255, cv2.THRESH_BINARY)
        outside_boundary = outside_boundary.astype(np.uint8)

        # draw contours fill the blank
        contours, _ = cv2.findContours(outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)

        # visualize the walls_skeleton_hist
        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "full_map1.png"), outside_boundary)

            # save the full map as point cloud
            positions = np.where(outside_boundary != 0)
            # switch the first 2 dimension of the positions
            positions = np.asarray(positions).T
            position_world = translate_grid_to_point(np.asarray(positions), grid_resolution=room_resolution,
                                                     voxel_dimension=room_dimension)
            # save the pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(position_world)
            o3d.io.write_point_cloud(os.path.join(tmp_floor_path, f"full_map_ori.ply"), pcd)

        # print(outside_boundary.shape, walls_skeleton_hist.shape)
        wall_positions = np.where(walls_skeleton_hist != 0)
        outside_boundary[wall_positions] = 0

        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "full_map2.png"), outside_boundary)

        # draw the outside boundary
        outside_boundary_tmp = cv2.bitwise_not(outside_boundary)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            outside_boundary_tmp, connectivity=8)

        # get all components with the area larger than 10
        outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 10:
                outside_boundary_tmp[labels == i] = 255

        outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)

        # visualize the largest component
        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "full_map3.png"), outside_boundary_tmp)

        # get the contours of the outside boundary
        contours, _ = cv2.findContours(outside_boundary_tmp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
        cv2.drawContours(outside_boundary_tmp, contours, -1, (255, 255, 255), -1)
        outside_boundary_tmp = outside_boundary_tmp.astype(np.uint8)

        # get the components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            outside_boundary_tmp, connectivity=8)
        # get the largest component
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
        outside_boundary_tmp[labels == max_label] = 255

        # apply closing to the outside boundary
        outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        outside_boundary_tmp = cv2.morphologyEx(outside_boundary_tmp,
                                                cv2.MORPH_CLOSE,
                                                kernal,
                                                iterations=2)
        outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)

        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "full_map4.png"), outside_boundary_tmp)
            #
            # plt.figure()
            # plt.imshow(outside_boundary_tmp, cmap="gray", origin="lower")
            # plt.savefig(os.path.join(tmp_floor_path, "2D_histogram_full_outside_contours.png"))

        # save the full map as point cloud
        positions = np.where(outside_boundary_tmp != 0)
        positions = np.asarray(positions).T
        position_world = translate_grid_to_point(np.asarray(positions),grid_resolution=room_resolution,voxel_dimension=room_dimension)
        # save the pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(position_world)
        o3d.io.write_point_cloud(os.path.join(tmp_floor_path, f"full_map.ply"), pcd)

        full_map = cv2.bitwise_not(outside_boundary_tmp)
        # # convert white to black and black to white
        # outside_boundary = cv2.bitwise_not(outside_boundary)
        # # dilate the outside boundary
        # kernel = np.ones((3, 3), np.uint8)  # 3x3 的膨胀核
        # outside_boundary = cv2.dilate(outside_boundary, kernel, iterations=1)
        # # convert white to black and black to white
        # outside_boundary = cv2.bitwise_not(outside_boundary)

        if save_intermediate_results:
            cv2.imwrite(os.path.join(tmp_floor_path, "full_map.png"), full_map)

            # # plot the full map
            # plt.figure()
            # plt.imshow(full_map, cmap="gray", origin="lower")
            # plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))

        boundary_mask = full_map.copy()

        # 记录已发现的不连通区域
        found_region_centers = []
        found_region_masks = []

        # 结构元素
        # kernel = np.ones((3, 3), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 迭代膨胀
        iteration = 0
        max_iterations = 100  # 设置最大迭代次数，避免无限循环
        flag = True
        while iteration < max_iterations and flag:
            # 膨胀边界
            new_boundary = cv2.dilate(boundary_mask, kernel, iterations=1)

            # # save the new boundary
            # plt.figure()
            # plt.imshow(new_boundary, cmap="gray", origin="lower")
            # os.makedirs(os.path.join(tmp_floor_path, "process"), exist_ok=True)
            # plt.savefig(os.path.join(tmp_floor_path, f"process/new_boundary_{iteration}.png"))
            # plt.close()

            # because the connect compoent detector detect white connect region, so we need to invert the color
            new_boundary_inv = cv2.bitwise_not(new_boundary)
            # calculate the connected components
            num_labels, new_labels, new_stats, centers = cv2.connectedComponentsWithStats(
                new_boundary_inv, connectivity=8)
            # print(num_labels)

            if num_labels >= 2:
                for label in range(0, num_labels):
                    # mask = (new_labels == label).astype(np.uint8) * 255
                    # plt.imshow(mask, cmap="gray")
                    # plt.title(f"Connected Component {label}")
                    # plt.show()
                    # print(new_stats[label, cv2.CC_STAT_AREA])
                    if 20 < new_stats[label, cv2.CC_STAT_AREA] < 900:
                        print(centers[label])
                        found_region_centers.append(centers[label])
                        # remove this region from the new boundary
                        new_boundary[new_labels == label] = 255

                        # store the mask
                        mask = (new_labels == label).astype(np.uint8) * 255
                        found_region_masks.append(mask)

            # for label in range(1, num_labels):
            #     if label != max_label and new_stats[label, cv2.CC_STAT_AREA] > 0:
            #         # 提取新的区域
            #         new_region = (new_labels == label).astype(np.uint8) * 255
            #         found_regions.append(new_region)

            # 更新边界
            boundary_mask = new_boundary

            # # 终止条件：如果没有新区域产生，则结束
            # if len(found_regions) == 0:
            #     break

            iteration += 1

            # if the whole image is white, then stop the iteration
            if np.all(boundary_mask == 255):
                flag = False

        # Create the marker image for the watershed algorithm
        markers = np.zeros_like(boundary_mask, dtype=np.int32)
        for i, mask in enumerate(found_region_masks):
            markers[mask == 255] = i + 1

        # Draw the background marker
        circle_radius = 1  # in pixels
        cv2.circle(markers, (3, 3), circle_radius, len(found_region_masks) + 1, -1)

        # plt.figure()
        # plt.imshow(markers, cmap="jet", origin="lower")
        # plt.savefig(os.path.join(tmp_floor_path, "water_start.png"))
        # plt.show()

        # Perform the watershed algorithm
        full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
        cv2.watershed(full_map, markers)

        # also draw the nodes on the map
        nodes_positions = []
        for node in self.nodes:
            pos_world = node.position
            pos_map = translate_point_to_grid(pos_world,grid_resolution=room_resolution,voxel_dimension=room_dimension)
            nodes_positions.append(pos_map[:2])

        if save_intermediate_results:
            room_segs = markers.copy()
            for pos in nodes_positions:
                room_segs[pos[0], pos[1]] = 0
            room_segs[room_segs==len(found_region_masks) + 1] = 0
            room_segs_to_save = save_grid_map(room_segs)
            cv2.imwrite(os.path.join(tmp_floor_path, "markers.png"), room_segs_to_save)

            # plt.figure()
            # plt.imshow(markers, cmap="jet", origin="lower")
            # plt.scatter([pos[0] for pos in nodes_positions], [pos[1] for pos in nodes_positions],
            #             c="r",
            #             label="Nodes")
            # plt.savefig(os.path.join(tmp_floor_path, "markers.png"))

        # 遍历每个唯一的 region_id，并创建 mask
        unique_regions = np.unique(markers)
        region_masks = {}
        position_worlds = {}
        position_maps = {}
        for region_id in unique_regions:
            if region_id in [-1, 0, unique_regions[-1]]:  # 跳过 watershed 线和背景
                continue
            mask = (markers == region_id).astype(np.uint8) * 255
            region_masks[region_id] = mask  # 存储每个区域的 mask
            # transform to world point cloud
            positions = np.where(mask == 255)
            positions = np.asarray(positions).T
            position_world = translate_grid_to_point(np.asarray(positions),grid_resolution=room_resolution,voxel_dimension=room_dimension)
            position_map = translate_point_to_grid(position_world,grid_resolution=self.grid_resolution,voxel_dimension=self.voxel_dimension)[:, :2]
            position_worlds[region_id] = position_world
            position_maps[region_id] = position_map

            # determine which points belong to this room region
            nodes_in_region = []
            for node_idx, nodes_position in enumerate(nodes_positions):
                if mask[nodes_position[0], nodes_position[1]] == 255:
                    nodes_in_region.append(self.nodes[node_idx])

            if len(nodes_in_region) != 0:
                # room_node = Room_node(nodes_in_region, position_world, len(self.room_nodes))
                for node in nodes_in_region:
                    node.room_idx = region_id

                # # save the pcd
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(position_world)
                # o3d.io.write_point_cloud(os.path.join(tmp_floor_path, f"region_{len(self.room_nodes)}.ply"), pcd)

                # self.room_nodes.append(room_node)
        for node in self.nodes:
            room_idx = node.room_idx
            node_idx = node.idx
            current_position = nodes_positions[node_idx]
            if room_idx == -1:
                # find the closest mask to this node
                node_distance_to_mask = {}
                for region_id, mask in region_masks.items():
                    positions = np.where(mask == 255)
                    distance = np.linalg.norm(np.array(positions).T - current_position, axis=1)
                    distance_min = np.min(distance)
                    node_distance_to_mask[region_id] = distance_min
                closest_region_id = min(node_distance_to_mask, key=node_distance_to_mask.get)
                node.room_idx = closest_region_id

        remaining_node_idxs = [node.idx for node in self.nodes]

        while remaining_node_idxs:
            node_idx = remaining_node_idxs[0]  # 取出当前节点
            node = self.nodes[node_idx]
            room_idx = node.room_idx

            # 获取该 room 内的所有节点
            nodes_in_region = [self.nodes[idx] for idx in remaining_node_idxs if self.nodes[idx].room_idx == room_idx]
            node_idxs_in_region = [node.idx for node in nodes_in_region]
            room_node = Room_node(nodes_in_region, position_worlds[room_idx], position_maps[room_idx], len(self.room_nodes))
            logger.info(f"Room {len(self.room_nodes)} has Nodes: {node_idxs_in_region}")

            for node_in_region in nodes_in_region:
                node_in_region.room_idx = len(self.room_nodes)

            self.room_nodes.append(room_node)

            # 移除已处理的节点
            remaining_node_idxs = list(set(remaining_node_idxs) - set(node_idxs_in_region))

        # update the distance from current pos to each room
        current_node = self.nodes[self.current_node_idx]
        for room_node in self.room_nodes:
            closet_node = self.find_closet_viewpoint_in_room(room_node)
            if closet_node is None:
                continue

            path_length = self.get_path_length(closet_node)
            room_node.distance = path_length


        # update the frontier state(frontier in room and frontier out room)
        # first get all the room mask
        room_map = np.ones((self.voxel_dimension[0], self.voxel_dimension[1]))
        for room_node in self.room_nodes:
            room_map_pos = room_node.mask_map
            room_map[room_map_pos[:, 0], room_map_pos[:, 1]] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        room_map = cv2.dilate(room_map, kernel, iterations=2)

        for frontier_idxs in self.current_global_frontier_map_idxs:
            # boarder_frontier = [[x,y] for (x,y) in frontier_idxs if room_map[x, y] == 1]
            inner_frontier = [[x,y] for (x,y) in frontier_idxs if room_map[x, y] == 0]

            if len(inner_frontier) > 0.6 * len(frontier_idxs):
                # print(inner_frontier)
                # print(frontier_idxs)
                # print(len(inner_frontier), len(frontier_idxs))
                self.grid_map[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1
            else:
                self.grid_map[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 2


    def find_closet_viewpoint_in_room(self, room_node):
        nodes = [node for node in room_node.nodes if node.state == 0 and node.has_frontier is True]
        if len(nodes) == 0:
            return None
        # get the positions of the nodes whose state is 0
        nodes_positions = [node.position for node in nodes]
        nodes_positions = np.array(nodes_positions)
        nodes_positions = nodes_positions + self.initial_position
        nodes_positions[:, 2] = self.initial_position[2] - 0.88
        nodes_positions = nodes_positions[:, [0, 2, 1]]

        current_position = self.current_position + self.initial_position
        current_position = np.array(
            [current_position[0], self.initial_position[2] - 0.88, current_position[1]])
        distance = [
            self.env.sim.geodesic_distance(current_position, node_position)
            for node_position in nodes_positions
        ]
        distance = np.array(distance)
        closet_node_idx = np.argmin(distance)
        closet_node = nodes[closet_node_idx]

        return closet_node

    def find_closest_nodes(self, nodes):
        nodes_positions = [node.position for node in nodes]
        nodes_positions = np.array(nodes_positions)
        nodes_positions = nodes_positions + self.initial_position
        nodes_positions[:, 2] = self.initial_position[2] - 0.88
        nodes_positions = nodes_positions[:, [0, 2, 1]]

        current_position = self.current_position + self.initial_position
        current_position = np.array(
            [current_position[0], self.initial_position[2] - 0.88, current_position[1]])
        distance = [
            self.env.sim.geodesic_distance(current_position, node_position)
            for node_position in nodes_positions
        ]
        distance = np.array(distance)
        closet_node_idx = np.argmin(distance)
        closet_node = nodes[closet_node_idx]

        return closet_node

    def explore_in_room(self, room_node):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        nodes = [node for node in room_node.nodes if node.has_frontier is True]
        if len(nodes) == 0:
            return None

        closet_node = self.find_closest_nodes(nodes)

        return closet_node


    def explore_in_room_relocate(self, room_node):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        nodes_true_frontier = [node for node in room_node.nodes if node.state == 0 and node.has_true_frontier is True]
        if len(nodes_true_frontier) == 0:
            logger.info('No true frontier in this room')
            inner_nodes = [node for node in room_node.nodes if node.has_frontier is True and node.state == 0 and node.has_true_frontier is False and node.frontier_idxs.shape[0] > self.frontier_thres]
            if len(inner_nodes) == 0:
                logger.info('No inner nodes in this room')
                room_node.state = 1
                for node in room_node.nodes:
                    node.has_frontier = False
                    node.has_true_frontier = False
                return None
            else:
                closet_node = self.find_closest_nodes(inner_nodes)

                return closet_node

        # may need improve
        closet_node = self.find_closest_nodes(nodes_true_frontier)

        return closet_node

    def explore_after_check(self):
        nodes_true_frontier = [node for node in self.nodes if node.state == 0 and node.has_frontier is True]
        if len(nodes_true_frontier) == 0:
            return None

        # may need improve
        closet_node = self.find_closest_nodes(nodes_true_frontier)

        return closet_node

    def explore_after_fully_explored(self):
        nodes = [node for node in self.nodes if node.state == 0]

        if len(nodes) == 0:
            return None

        closet_node = self.find_closest_nodes(nodes)

        return closet_node


    def get_path_length(self, end):

        # return np.array([end.position]), [end.idx]

        path_node_position, path_node_idx = self.find_the_closest_path(self.current_position, end.position)
        path_node_position = path_node_position[1:]
        path_node_idx = path_node_idx[1:]
        coeff = 1.0
        penal_coeff = 1.1 * (1 + 0.5 * self.update_iterations/500.0)
        for node_idx in path_node_idx:
            node = self.nodes[node_idx]
            if node.state == NodeState.EXPLORED:
                coeff = penal_coeff * coeff

        # calculate the geometric distance from the current position to the end position
        current_position = self.current_position
        current_position = current_position + self.initial_position
        current_position[2] = self.initial_position[2]-0.88
        current_position = [current_position[0], current_position[2], current_position[1]]
        end_position = end.position + self.initial_position
        end_position[2] = self.initial_position[2]-0.88
        end_position = [end_position[0], end_position[2], end_position[1]]
        distance_to_end = self.env.sim.geodesic_distance(current_position, end_position)

        distance_to_end = coeff * distance_to_end

        return distance_to_end

    def to_json_save_node_info(self):
        json_data = {}
        object_json = []
        for i in range(len(self.objects)):
            obj = self.objects[i]
            nwpcd = obj.pcd.point.positions.cpu().numpy()
            size = np.zeros(3)
            size[0] = np.max(nwpcd[:, 0]) - np.min(nwpcd[:, 0])
            size[1] = np.max(nwpcd[:, 1]) - np.min(nwpcd[:, 1])
            size[2] = np.max(nwpcd[:, 2]) - np.min(nwpcd[:, 2])

            object_json.append({
                'index': i,
                'position': [round(p, 3) for p in obj.position.tolist()],
                'class': obj.tag,
                'confidence': round(obj.confidence.numpy().item(), 3),
                'size': [round(s, 3) for s in size.tolist()],
            })
        json_data['objects'] = object_json

        room_nodes = []
        for i in range(len(self.room_nodes)):
            room_node = self.room_nodes[i]
            nodes_in_room = room_node.nodes
            nodes = []
            for j in range(len(nodes_in_room)):
                node = nodes_in_room[j]
                if node.state == NodeState.EXPLORED:
                    node.has_frontier = False
                    node.has_true_frontier = False
                nodes.append({
                    'idx': node.idx,
                    'position': [round(p, 3) for p in node.position.tolist()],
                    'state': node.state,
                    'neighbors': self.neighbors[node.idx],
                    'has_frontier': node.has_frontier,
                    'has_true_frontier': node.has_true_frontier,
                })

                nodes[-1]['objects'] = node.objects
            room_nodes.append({
                'room_idx': room_node.room_id,
                'state': room_node.state,
                'distance': round(room_node.distance, 3),
                'viewpoints': nodes,
            })

        json_data['Room'] = room_nodes

        return json_data