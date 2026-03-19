import open3d as o3d
import numpy as np


def create_digit_point_cloud(digit):
    """
    根据数字创建点云。
    :param digit: 数字 (0-9)
    :return: Open3D PointCloud 对象
    """
    if digit not in range(0, 10):
        raise ValueError("Only digits 0-9 are supported.")

    # 定义每个数字的点坐标
    digit_coords = {
        0: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(0, y) for y in np.linspace(1, 0, 100)] +  # 左竖线
           [(1, y) for y in np.linspace(1, 0, 100)] +  # 右竖线
           [(x, 0) for x in np.linspace(0, 1, 50)],  # 底部横线
        1: [(0, y) for y in np.linspace(0, 1, 100)],
        2: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(1, y) for y in np.linspace(1, 0.5, 50)] +  # 右上竖线
           [(x, 0.5) for x in np.linspace(1, 0, 50)] +  # 中间横线
           [(0, y) for y in np.linspace(0.5, 0, 50)] +  # 左下竖线
           [(x, 0) for x in np.linspace(0, 1, 50)],  # 底部横线
        3: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(1, y) for y in np.linspace(1, 0.5, 50)] +  # 右上竖线
           [(x, 0.5) for x in np.linspace(1, 0, 50)] +  # 中间横线
           [(1, y) for y in np.linspace(0.5, 0, 50)] +  # 右下竖线
           [(x, 0) for x in np.linspace(1, 0, 50)],  # 底部横线
        4: [(0, y) for y in np.linspace(1, 0.5, 50)] +  # 左上竖线
           [(1, y) for y in np.linspace(1, 0, 100)] +  # 右竖线
           [(x, 0.5) for x in np.linspace(0, 1, 50)],  # 中间横线
        5: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(0, y) for y in np.linspace(1, 0.5, 50)] +  # 左上竖线
           [(x, 0.5) for x in np.linspace(0, 1, 50)] +  # 中间横线
           [(1, y) for y in np.linspace(0.5, 0, 50)] +  # 右下竖线
           [(x, 0) for x in np.linspace(1, 0, 50)],  # 底部横线
        6: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(0, y) for y in np.linspace(1, 0, 100)] +  # 左竖线
           [(x, 0.5) for x in np.linspace(0, 1, 50)] +  # 中间横线
           [(1, y) for y in np.linspace(0.5, 0, 50)] +  # 右下竖线
           [(x, 0) for x in np.linspace(1, 0, 50)],  # 底部横线
        7: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(1, y) for y in np.linspace(1, 0, 100)],  # 右竖线
        8: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(0, y) for y in np.linspace(1, 0, 100)] +  # 左竖线
           [(1, y) for y in np.linspace(1, 0, 100)] +  # 右竖线
           [(x, 0.5) for x in np.linspace(0, 1, 50)] +  # 中间横线
           [(x, 0) for x in np.linspace(0, 1, 50)],  # 底部横线
        9: [(x, 1) for x in np.linspace(0, 1, 50)] +  # 顶部横线
           [(0, y) for y in np.linspace(1, 0.5, 50)] +  # 左上竖线
           [(x, 0.5) for x in np.linspace(0, 1, 50)] +  # 中间横线
           [(1, y) for y in np.linspace(1, 0, 100)],  # 右竖线
    }

    # 获取点云坐标
    coords = digit_coords[digit]
    points = np.array([[x, y, 0] for x, y in coords])  # z 坐标为 0

    # 创建 Open3D PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_number_point_cloud(number, center):
    """
    根据任意数字创建点云，并设置中心点。
    :param number: 任意位数的数字
    :param center: 3D 坐标，表示点云的中心点
    :return: Open3D PointCloud 对象
    """
    digits = [int(d) for d in str(number)]  # 将数字拆分为各个位数

    combined_points = []
    offset = 0
    for digit in digits:
        pcd = create_digit_point_cloud(digit)
        points = np.asarray(pcd.points)
        points[:, 0] += offset  # 设置 X 方向的偏移量
        combined_points.append(points)
        offset += 1  # 每个数字之间的间隔

    # 合并所有点
    all_points = np.vstack(combined_points)

    # scale all the points to 0.05
    all_points = all_points * 0.1

    # 平移点云到中心点
    all_points += np.array(center) - np.mean(all_points, axis=0)

    # 向右平移0.05
    all_points[:, 0] += 0.3

    # 创建最终的点云
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_points)
    #paint all the points to green
    colors = np.array([[0, 255, 0] for _ in range(len(all_points))])
    final_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    return final_pcd
