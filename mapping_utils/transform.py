import numpy as np
import quaternion


def habitat_camera_intrinsic(config):
    camera_config = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    # camera_config = config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor

    assert camera_config.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert camera_config.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert camera_config.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = camera_config.width
    height = camera_config.height
    hfov = camera_config.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f, 0, xc], [0, f, zc], [0, 0, 1]], np.float32)
    return intrinsic_matrix


def habitat_translation(position):
    return np.array([position[0], position[2], position[1]])


def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    rotation_matrix = np.matmul(transform_matrix, rotation_matrix)
    return rotation_matrix
