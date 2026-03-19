import os

import habitat
from habitat.config.default_structured_configs import (CollisionsMeasurementConfig,
                                                       FogOfWarConfig,
                                                       TopDownMapMeasurementConfig)
from habitat.config.read_write import read_write


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


HABITAT_LAB_PATH = _get_env("HABITAT_LAB_PATH")
HM3D_CONFIG_PATH = os.path.join(
    HABITAT_LAB_PATH,
    "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
MP3D_CONFIG_PATH = os.path.join(
    HABITAT_LAB_PATH,
    "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")

SAM_CHECKPOINT = _get_env("SAM_CHECKPOINT")
GROUNDING_DINO_PATH = _get_env("GROUNDING_DINO_PATH")
GROUNDING_DINO_CONFIG = os.path.join(
    GROUNDING_DINO_PATH,
    "configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py")
GROUNDING_DINO_CHECKPOINT = _get_env("GROUNDING_DINO_CHECKPOINT")


def hm3d_config(path: str = HM3D_CONFIG_PATH, stage: str = 'val', episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        data_path = _get_env("HM3D_DATA_PATH")
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(
            data_path, "scene_datasets")
        habitat_config.habitat.dataset.data_path = os.path.join(
            data_path, "objectnav_hm3d_v2/{split}/{split}.json.gz")
        habitat_config.habitat.simulator.scene_dataset = os.path.join(
            data_path,
            "scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json")
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.iterator_options.shuffle = False
        habitat_config.habitat.environment.iterator_options.group_by_scene = False
        habitat_config.habitat.task.measurements.update({
            "top_down_map":
                TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
            "collisions":
                CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 1.0

        # habitat_config.habitat.task.measurements.top_down_map.draw_goal_aabbs = False
        # habitat_config.habitat.task.measurements.top_down_map.draw_goal_positions = False
        # habitat_config.habitat.task.measurements.top_down_map.draw_view_points = False

        habitat_config.habitat.environment.max_episode_steps = 500

    return habitat_config


def mp3d_config(path: str = MP3D_CONFIG_PATH, stage: str = 'val', episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        data_path = _get_env("MP3D_DATA_PATH")
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(
            data_path, "scene_datasets")
        habitat_config.habitat.dataset.data_path = os.path.join(
            data_path, "habitat_task/objectnav/{split}/{split}.json.gz")
        habitat_config.habitat.simulator.scene_dataset = os.path.join(
            data_path, "scene_datasets/config.json")
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.iterator_options.shuffle = False
        habitat_config.habitat.environment.iterator_options.group_by_scene = False
        habitat_config.habitat.task.measurements.update({
            "top_down_map":
                TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
            "collisions":
                CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.task.measurements.success.success_distance = 1.0

        # habitat_config.habitat.task.measurements.top_down_map.draw_goal_aabbs = False
        # habitat_config.habitat.task.measurements.top_down_map.draw_goal_positions = False
        # habitat_config.habitat.task.measurements.top_down_map.draw_view_points = False

        habitat_config.habitat.environment.max_episode_steps = 500

    return habitat_config
