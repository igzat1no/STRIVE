import numpy as np


def calc_depth_img(camera_intrinsic, min_depth, max_depth):
    """
    计算最小深度和最大深度球面投影到深度图上的z轴深度值

    参数:
    - camera_intrinsic: (3,3) numpy 数组, 内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    - min_depth: 到相机光心的最小球面深度
    - max_depth: 到相机光心的最大球面深度

    返回:
    - min_depth_image: (H, W) 深度图，对应 min_depth 球面
    - max_depth_image: (H, W) 深度图，对应 max_depth 球面
    """
    w, h = 640, 480  # 图像尺寸

    # fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    # cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    #
    # # 构建像素网格
    # u, v = np.meshgrid(np.arange(w), np.arange(h))
    #
    # # 计算归一化坐标 (X, Y) = ((u - cx) / fx, (v - cy) / fy)
    # x = (u - cx) / fx
    # y = (v - cy) / fy
    #
    # # 计算每个像素点的 Z 轴深度
    # def compute_depth_image(sphere_depth):
    #     return np.sqrt(sphere_depth**2 / (1 + x**2 + y**2))  # 由球面方程计算 Z 轴分量
    #
    # min_depth_image = compute_depth_image(min_depth)
    # max_depth_image = compute_depth_image(max_depth)

    min_depth_image = np.ones((h, w), np.float32) * min_depth
    max_depth_image = np.ones((h, w), np.float32) * max_depth

    return min_depth_image, max_depth_image


def ori_preprocess_depth(depth: np.ndarray,
                         lower_bound: float = 0.51,
                         upper_bound: float = 4.9):
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    for i in range(depth.shape[1]):
        depth_line = depth[:, i]
        if len(depth_line[depth_line > 0]) == 0:
            continue
        indexs = [[-1, -1]]
        values = [0]
        # print(depth_line)
        for j in range(len(depth_line) - 2, int(len(depth_line) * 0.5 - 1), -1):
            if depth_line[j - 1] < depth_line[j] - 0.08 and depth_line[j - 1] > 0.58:
                # print(f'Diff: {depth_line[j] - depth_line[j - 1]}')
                # find the index of the element closet to depth_line[j-1] in depth_line[j:]
                # closest_index = np.argmin(np.abs(depth_line[j:indexs[-1][0]] - depth_line[j - 1])) + j
                closest_index = np.argmin(np.abs(depth_line[j:] - depth_line[j - 1])) + j
                # print(depth_line[j:indexs[-1][0]])
                # print(depth_line[j - 1])
                indexs.append([j, closest_index])
                values.append(depth_line[j - 1])

        indexs = indexs[1:]
        values = values[1:]
        # print(indexs)
        # print(values)
        if len(indexs) == 0:
            continue
        for [start, end], value in zip(indexs, values):
            if start == end:
                continue
            for idx in range(start, end + 1):
                if depth[idx, i] > value:
                    depth[idx, i] = value

    depth[np.where((depth < lower_bound) | (depth > upper_bound))] = 0
    return depth


# def preprocess_depth(depth:np.ndarray,lower_img, upper_img):
#     if len(depth.shape) == 3:
#         depth = depth[:,:,0]
#
#     # for i in range(depth.shape[1]):
#     #     depth_line = depth[:, i]
#     #     if len(depth_line[depth_line > 0]) == 0:
#     #         continue
#     #     indexs = [[-1, -1]]
#     #     values = [0]
#     #     # print(depth_line)
#     #     for j in range(len(depth_line) - 2, int(len(depth_line) * 0.5 - 1), -1):
#     #         if depth_line[j - 1] < depth_line[j] - 0.08 and depth_line[j - 1] > 0.58:
#     #             # print(f'Diff: {depth_line[j] - depth_line[j - 1]}')
#     #             # find the index of the element closet to depth_line[j-1] in depth_line[j:]
#     #             # closest_index = np.argmin(np.abs(depth_line[j:indexs[-1][0]] - depth_line[j - 1])) + j
#     #             closest_index = np.argmin(np.abs(depth_line[j:] - depth_line[j - 1])) + j
#     #             # print(depth_line[j:indexs[-1][0]])
#     #             # print(depth_line[j - 1])
#     #             indexs.append([j, closest_index])
#     #             values.append(depth_line[j - 1])
#     #
#     #     indexs = indexs[1:]
#     #     values = values[1:]
#     #     # print(indexs)
#     #     # print(values)
#     #     if len(indexs) == 0:
#     #         continue
#     #     for [start, end], value in zip(indexs, values):
#     #         if start == end: continue
#     #         for idx in range(start, end+1):
#     #             if depth[idx, i] > value:
#     #                 depth[idx, i] = value
#
#     depth[depth < lower_img] = 0
#     depth[depth > upper_img] = 0
#
#     return depth


def preprocess_depth(depth: np.ndarray,
                     lower_bound: float = 0.51,
                     upper_bound: float = 4.9):
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # for i in range(depth.shape[1]):
    #     depth_line = depth[:, i]
    #     if len(depth_line[depth_line > 0]) == 0:
    #         continue
    #     indexs = [[-1, -1]]
    #     values = [0]
    #     # print(depth_line)
    #     for j in range(len(depth_line) - 2, int(len(depth_line) * 0.5 - 1), -1):
    #         if depth_line[j - 1] < depth_line[j] - 0.08 and depth_line[j - 1] > 0.58:
    #             # print(f'Diff: {depth_line[j] - depth_line[j - 1]}')
    #             # find the index of the element closet to depth_line[j-1] in depth_line[j:]
    #             # closest_index = np.argmin(np.abs(depth_line[j:indexs[-1][0]] - depth_line[j - 1])) + j
    #             closest_index = np.argmin(np.abs(depth_line[j:] - depth_line[j - 1])) + j
    #             # print(depth_line[j:indexs[-1][0]])
    #             # print(depth_line[j - 1])
    #             indexs.append([j, closest_index])
    #             values.append(depth_line[j - 1])
    #
    #     indexs = indexs[1:]
    #     values = values[1:]
    #     # print(indexs)
    #     # print(values)
    #     if len(indexs) == 0:
    #         continue
    #     for [start, end], value in zip(indexs, values):
    #         if start == end: continue
    #         for idx in range(start, end+1):
    #             if depth[idx, i] > value:
    #                 depth[idx, i] = value

    depth[np.where((depth < lower_bound) | (depth > upper_bound))] = 0

    return depth


def preprocess_image(image: np.ndarray):
    return image
