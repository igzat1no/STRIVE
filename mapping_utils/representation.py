import numpy as np
from enum import IntEnum
import heapq
import json


class NodeState(IntEnum):
    UNEXPLORED = 0
    EXPLORED = 1


class ObjectNode:

    def __init__(self, pcd_all, pcd_filtered, tag, confidence, position, actual_num, rgb,
                 bbox):
        self.pcd_all = pcd_all
        self.pcd = pcd_filtered
        self.tag = tag
        self.confidence = confidence
        self.position = position
        self.nodes = []
        self.num_list = {tag: actual_num}
        self.conf_list = {tag: confidence}
        self.alive = True
        self.rgb = rgb
        self.bbox = bbox

    def find_closest(self, pos):
        pts = self.pcd.point.positions.cpu().numpy()
        dist = np.linalg.norm(pts - pos, axis=1)
        idx = np.argmin(dist)
        return pts[idx]
        # return self.position


class our_Node:

    def __init__(self,
                 rgb_img,
                 depth_img,
                 pcd,
                 position,
                 encoder,
                 idx,
                 has_frontier=False,
                 frontier_idxs=np.array([]).reshape((-1, 2))):
        self.state = NodeState.UNEXPLORED
        # self.rgb_imgs = [rgb_img]
        # self.depth_imgs = [depth_img]
        self.pcd = pcd
        self.position = position
        self.idx = idx
        self.objects = []
        self.has_frontier = has_frontier
        self.has_true_frontier = False
        # self.rgb_features = [encoder.encode_rgb(rgb_img)]
        # self.depth_features = [encoder.encode_depth(depth_img)]
        self.room_idx = None
        self.frontier_idxs = frontier_idxs.reshape((-1, 2))

    def update(self, rgb_img, depth_img, pointcloud, encoder=None):
        self.rgb_imgs = [rgb_img]
        self.depth_imgs = [depth_img]
        self.rgb_features = []
        self.depth_features = []
        if encoder is not None:
            self.rgb_features = [encoder.encode_rgb(rgb_img)]
            self.depth_features = [encoder.encode_depth(depth_img)]
        self.pointcloud = pointcloud

    def update_obj(self, obj_indices):
        self.objects += obj_indices
        self.objects = list(set(self.objects))

    def upgrade(self, rgb_imgs, depth_imgs, encoder):
        self.state = NodeState.EXPLORED
        self.rgb_imgs = rgb_imgs
        self.depth_imgs = depth_imgs
        self.rgb_features = []
        self.depth_features = []
        for rgb_img in rgb_imgs:
            self.rgb_features.append(encoder.encode_rgb(rgb_img))
        for depth_img in depth_imgs:
            self.depth_features.append(encoder.encode_depth(depth_img))

    # def combine_unexplored(self, node):
    #     assert node.state == NodeState.UNEXPLORED
    #     num1 = len(self.rgb_imgs)
    #     num2 = len(node.rgb_imgs)
    #     self.position = (num1 * self.position + num2 * node.position) / (num1 + num2)
    #     self.rgb_imgs += node.rgb_imgs
    #     self.depth_imgs += node.depth_imgs
    #     self.rgb_features += node.rgb_features
    #     self.depth_features += node.depth_features


class Room_node:

    def __init__(self, nodes, mask, mask_map, room_id):
        self.nodes = nodes
        self.mask = mask
        self.mask_map = mask_map
        self.room_id = room_id

        # if len(nodes) == 0:
        #     self.state = 1
        #     self.nodes_idx = []
        # else:
        self.nodes_idx = [node.idx for node in nodes]

        self.state = 1
        for node in self.nodes:
            if node.has_frontier:
                self.state = 0
                break

        self.distance = 1e5

    def update_state(self):
        self.state = 1
        for node in self.nodes:
            if node.has_true_frontier:
                self.state = 0
                break
