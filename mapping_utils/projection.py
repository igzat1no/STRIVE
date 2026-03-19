import numpy as np
import open3d as o3d
import cv2
# obstacle = 0
# unknown = 1
# position = 2
# navigable = 3
# frontier = 4


def project_frontier(obstacle_pcd,
                     navigable_pcd,
                     obstacle_height=-0.7,
                     grid_resolution=0.1,
                     voxel_dimension=[500, 500, 20]):
    np_obstacle_points = obstacle_pcd.point.positions.cpu().numpy()
    np_navigable_points = navigable_pcd.point.positions.cpu().numpy()
    grid_dimensions = np.array(voxel_dimension).astype(int)
    grid_map = np.ones((grid_dimensions[0], grid_dimensions[1]), dtype=np.int32)
    # get navigable occupancy
    navigable_points = np_navigable_points
    navigable_indices = translate_point_to_grid(navigable_points, grid_resolution,
                                                grid_dimensions)
    navigable_indices[:, 0] = np.clip(navigable_indices[:, 0], 0, grid_dimensions[0] - 1)
    navigable_indices[:, 1] = np.clip(navigable_indices[:, 1], 0, grid_dimensions[1] - 1)
    navigable_indices[:, 2] = np.clip(navigable_indices[:, 2], 0, grid_dimensions[2] - 1)
    navigable_voxels = np.zeros(grid_dimensions, dtype=np.int32)
    navigable_voxels[navigable_indices[:, 0], navigable_indices[:, 1],
                     navigable_indices[:, 2]] = 1
    navigable_map = (navigable_voxels.sum(axis=2) > 0)
    grid_map[np.where(navigable_map > 0)] = 3
    # get obstacle occupancy
    obstacle_points = np_obstacle_points
    obstacle_indices = translate_point_to_grid(obstacle_points, grid_resolution,
                                               grid_dimensions)
    obstacle_indices[:, 0] = np.clip(obstacle_indices[:, 0], 0, grid_dimensions[0] - 1)
    obstacle_indices[:, 1] = np.clip(obstacle_indices[:, 1], 0, grid_dimensions[1] - 1)
    obstacle_indices[:, 2] = np.clip(obstacle_indices[:, 2], 0, grid_dimensions[2] - 1)
    obstacle_voxels = np.zeros(grid_dimensions, dtype=np.int32)
    obstacle_voxels[obstacle_indices[:, 0], obstacle_indices[:, 1],
                    obstacle_indices[:, 2]] = 1
    obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
    grid_map[np.where(obstacle_map > 0)] = 0
    # get outer-border of navigable areas
    outer_border_navigable = ((grid_map == 3) * 255).astype(np.uint8)
    contours, hierarchiy = cv2.findContours(outer_border_navigable, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    outer_border_navigable = cv2.drawContours(
        np.zeros((grid_map.shape[0], grid_map.shape[1])), contours, -1, (255, 255, 255),
        1).astype(np.float32)
    obstacles = ((grid_map == 0) * 255).astype(np.float32)
    obstacles = cv2.dilate(obstacles.astype(np.uint8), np.ones((3, 3)))
    # obstacles = cv2.dilate(obstacles.astype(np.uint8),np.ones((4,4)))
    outer_border_navigable = ((outer_border_navigable - obstacles) > 0)
    grid_map_x, grid_map_y = np.where(outer_border_navigable > 0)
    z_value = -0.8 // grid_resolution + grid_dimensions[2] // 2 - 1
    grid_indexes = np.stack((grid_map_x, grid_map_y, z_value * np.ones(
        (grid_map_x.shape[0],))),
                            axis=1)
    frontier_points = translate_grid_to_point(grid_indexes, grid_resolution,
                                              grid_dimensions)

    frontier_indexs = [grid_map_x, grid_map_y]
    frontier_map = np.zeros_like(grid_map)
    frontier_map[grid_map_x, grid_map_y] = 1

    return frontier_map, frontier_points


def project_frontier_and_obstacle(obstacle_pcd,
                                  navigable_pcd,
                                  obstacle_height=-0.7,
                                  grid_resolution=0.25):
    np_obstacle_points = obstacle_pcd.point.positions.cpu().numpy()
    np_navigable_points = navigable_pcd.point.positions.cpu().numpy()
    np_all_points = np.concatenate((np_obstacle_points, np_navigable_points), axis=0)
    max_bound = np.max(np_all_points, axis=0)
    min_bound = np.min(np_all_points, axis=0)
    grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    grid_map = np.ones((grid_dimensions[0], grid_dimensions[1]), dtype=np.int32)
    # get navigable occupancy
    navigable_points = np_navigable_points
    navigable_indices = np.floor(
        (navigable_points - min_bound) / grid_resolution).astype(int)
    navigable_indices[:, 0] = np.clip(navigable_indices[:, 0], 0, grid_dimensions[0] - 1)
    navigable_indices[:, 1] = np.clip(navigable_indices[:, 1], 0, grid_dimensions[1] - 1)
    navigable_indices[:, 2] = np.clip(navigable_indices[:, 2], 0, grid_dimensions[2] - 1)
    navigable_voxels = np.zeros(grid_dimensions, dtype=np.int32)
    navigable_voxels[navigable_indices[:, 0], navigable_indices[:, 1],
                     navigable_indices[:, 2]] = 1
    navigable_map = (navigable_voxels.sum(axis=2) > 0)
    grid_map[np.where(navigable_map > 0)] = 3
    # get obstacle occupancy
    obstacle_points = np_obstacle_points
    obstacle_indices = np.floor(
        (obstacle_points - min_bound) / grid_resolution).astype(int)
    obstacle_indices[:, 0] = np.clip(obstacle_indices[:, 0], 0, grid_dimensions[0] - 1)
    obstacle_indices[:, 1] = np.clip(obstacle_indices[:, 1], 0, grid_dimensions[1] - 1)
    obstacle_indices[:, 2] = np.clip(obstacle_indices[:, 2], 0, grid_dimensions[2] - 1)
    obstacle_voxels = np.zeros(grid_dimensions, dtype=np.int32)
    obstacle_voxels[obstacle_indices[:, 0], obstacle_indices[:, 1],
                    obstacle_indices[:, 2]] = 1
    obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
    grid_map[np.where(obstacle_map > 0)] = 0
    # get outer-border of navigable areas
    outer_border_navigable = ((grid_map == 3) * 255).astype(np.uint8)
    contours, hierarchiy = cv2.findContours(outer_border_navigable, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    outer_border_navigable = cv2.drawContours(
        np.zeros((grid_map.shape[0], grid_map.shape[1])), contours, -1, (255, 255, 255),
        1).astype(np.float32)

    obstacles = ((grid_map == 0) * 255).astype(np.float32)
    obstacles = cv2.dilate(obstacles.astype(np.uint8), np.ones((3, 3)))
    outer_border_navigable = ((outer_border_navigable - obstacles) > 0)
    grid_map_x, grid_map_y = np.where(outer_border_navigable > 0)
    grid_indexes = np.stack((grid_map_x, grid_map_y, obstacle_height * np.ones(
        (grid_map_x.shape[0],))),
                            axis=1)
    frontier_points = grid_indexes * grid_resolution + min_bound

    outer_border_navigable = cv2.dilate(outer_border_navigable.astype(np.uint8),
                                        np.ones((3, 3)))
    obstacles_in_scene = ((obstacles - outer_border_navigable) > 0)
    grid_map_x, grid_map_y = np.where(obstacles_in_scene > 0)
    grid_indexes = np.stack((grid_map_x, grid_map_y, obstacle_height * np.ones(
        (grid_map_x.shape[0],))),
                            axis=1)
    obstacle_points = grid_indexes * grid_resolution + min_bound

    return frontier_points, obstacle_points


def translate_single_grid_to_point(grid_index,
                                   grid_resolution=0.1,
                                   voxel_dimension=[500, 500, 20]):
    # print(grid_indexes.shape)
    if grid_index.shape[0] == 2:
        z_value = -0.8 // grid_resolution + voxel_dimension[2] // 2 - 1
        grid_index = np.array([grid_index[0], grid_index[1], z_value])

    voxel_dimension = np.array(voxel_dimension)
    np_all_points = (grid_index - voxel_dimension // 2 + 1) * grid_resolution

    return np_all_points


def translate_grid_to_point(grid_indexes,
                            grid_resolution=0.1,
                            voxel_dimension=[500, 500, 20]):
    # print(grid_indexes.shape)
    if grid_indexes.shape[1] == 2:
        z_value = -0.8 // grid_resolution + voxel_dimension[2] // 2 - 1
        grid_indexes = np.stack(
            (grid_indexes[:, 0], grid_indexes[:, 1], z_value * np.ones(
                (grid_indexes.shape[0],))),
            axis=1)

    voxel_dimension = np.array(voxel_dimension)
    np_all_points = (grid_indexes - voxel_dimension // 2 + 1) * grid_resolution

    return np_all_points


def translate_single_point_to_grid(points,
                                   grid_resolution=0.1,
                                   voxel_dimension=[500, 500, 20]):
    # project the pointcloud to a 500x500x20 voxel map
    voxel_dimension = np.array(voxel_dimension)
    np_all_points = points
    grid_index = (np.floor(np_all_points[:] / grid_resolution).astype(int) +
                  voxel_dimension // 2 - 1).astype(int)

    return grid_index


def translate_point_to_grid(points, grid_resolution=0.1, voxel_dimension=[500, 500, 20]):
    # project the pointcloud to a 500x500x20 voxel map
    voxel_dimension = np.array(voxel_dimension)
    np_all_points = points
    grid_index = (np.floor(np_all_points[:] / grid_resolution).astype(int) +
                  voxel_dimension // 2 - 1).astype(int)

    return grid_index


def project_room(navigable_points,
                 grid_resolution=0.05,
                 voxel_dimension=[1000, 1000, 20]):
    grid_dimensions = np.array(voxel_dimension).astype(int)
    # grid_resolution = 0.05
    navigable_indices = translate_point_to_grid(navigable_points, grid_resolution,
                                                grid_dimensions)

    navigable_indices[:, 0] = np.clip(navigable_indices[:, 0], 0, grid_dimensions[0] - 1)
    navigable_indices[:, 1] = np.clip(navigable_indices[:, 1], 0, grid_dimensions[1] - 1)
    navigable_indices[:, 2] = np.clip(navigable_indices[:, 2], 0, grid_dimensions[2] - 1)
    navigable_voxels = np.zeros(grid_dimensions, dtype=np.int32)
    for point in navigable_indices:
        navigable_voxels[point[0], point[1], point[2]] += 1
    # navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = 1
    navigable_map = navigable_voxels.sum(axis=2).astype(np.float32)

    # print(navigable_map.shape)

    return navigable_map


def save_grid_map(grid_map):
    # switch the x and y axis
    grid_map = np.transpose(grid_map)
    # flip the y axis
    grid_map = np.flip(grid_map, 0)
    # assign color to each unique value in the grid map
    unique_values = np.unique(grid_map)
    color_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
    for i in range(unique_values.shape[0]):
        if unique_values[i] != 0 and unique_values[i] != -1:
            color_map[np.where(grid_map == unique_values[i])] = np.random.randint(
                0, 255, (1, 3))
        else:
            color_map[np.where(grid_map == unique_values[i])] = [0, 0, 0]

    return color_map


def bresenham_3d(p1, p2):
    """
    使用Bresenham算法获取3D网格中两点之间的所有voxel。

    参数:
        p1: tuple(int, int, int)，起点 (x1, y1, z1)
        p2: tuple(int, int, int)，终点 (x2, y2, z2)

    返回:
        list of tuple(int, int, int)：连线经过的所有voxel坐标
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    voxels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        py = 2 * dy - dx
        pz = 2 * dz - dx
        while x1 != x2:
            voxels.append((x1, y1, z1))
            x1 += xs
            if py >= 0:
                y1 += ys
                py -= 2 * dx
            if pz >= 0:
                z1 += zs
                pz -= 2 * dx
            py += 2 * dy
            pz += 2 * dz

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        px = 2 * dx - dy
        pz = 2 * dz - dy
        while y1 != y2:
            voxels.append((x1, y1, z1))
            y1 += ys
            if px >= 0:
                x1 += xs
                px -= 2 * dy
            if pz >= 0:
                z1 += zs
                pz -= 2 * dy
            px += 2 * dx
            pz += 2 * dz

    # Driving axis is Z-axis
    else:
        px = 2 * dx - dz
        py = 2 * dy - dz
        while z1 != z2:
            voxels.append((x1, y1, z1))
            z1 += zs
            if px >= 0:
                x1 += xs
                px -= 2 * dz
            if py >= 0:
                y1 += ys
                py -= 2 * dz
            px += 2 * dx
            py += 2 * dy

    voxels.append((x2, y2, z2))
    return voxels
