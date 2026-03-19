import argparse
import csv
import json
import os
import sys

import habitat
from loguru import logger
from matplotlib.pyplot import savefig
from tqdm import tqdm

from config_utils import hm3d_config, mp3d_config
from constants import *
from cv_utils.gpt_utils import ask_gpt_similar_objects
from mapper_with_process_obs import Instruct_Mapper
from mapping_utils.transform import habitat_camera_intrinsic
from objnav_agent_with_process_obs import HM3D_Objnav_Agent

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--mapper_resolution", type=float, default=0.05)
    parser.add_argument("--grid_resolution", type=float, default=0.1)
    parser.add_argument("--grid_size", type=int, default=500)
    parser.add_argument("--grid_height", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="default")
    parser.add_argument("--not_do_seg", default=True, action="store_false")
    parser.add_argument("--no_gpt_seg", default=True, action="store_false")
    parser.add_argument("--relocate", default=False, action="store_true")
    parser.add_argument("--no_gpt_relocate", default=False, action="store_true")
    parser.add_argument("--vlm", type=str, default="gemini")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    args.save_dir = "logs/" + args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
    habitat_env = habitat.Env(habitat_config)
    habitat_mapper = Instruct_Mapper(
        habitat_camera_intrinsic(habitat_config),
        pcd_resolution=args.mapper_resolution,
        grid_resolution=args.grid_resolution,
        voxel_dimension=[args.grid_size, args.grid_size, args.grid_height],
        save_dir=args.save_dir,
        categories=categories,
        no_gpt_seg=args.no_gpt_seg,
        env=habitat_env,
        vlm=args.vlm)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,
                                      habitat_mapper,
                                      save_dir=args.save_dir,
                                      do_seg=args.not_do_seg,
                                      relocate=args.relocate,
                                      gpt_relocate=not args.no_gpt_relocate,
                                      vlm=args.vlm)
    evaluation_metrics = []

    start_idx = args.start_episode
    episodes = habitat_agent.env.episode_iterator.episodes
    habitat_agent.env.reset()

    for i in tqdm(range(start_idx, args.eval_episodes)):
        # try:
        # logger_path = f"./{args.save_dir}/episode-{i}/log.txt"
        # logger.add(logger_path, mode="w")
        logger.info(f"Processing episode: i = {i}")

        habitat_agent.env.episode_iterator.set_next_episode_by_index(i)
        habitat_agent.reset(i)

        target = habitat_agent.instruct_goal
        start = target.find("<") + 1
        end = target.find(">")
        habitat_mapper.target = target[start:end]

        habitat_mapper.target_list = ask_gpt_similar_objects(
            habitat_mapper.object_perceiver.classes,
            habitat_mapper.target,
            args.vlm
        )

        logger.info(f"Target: {habitat_mapper.target}")
        logger.info(f"Target list: {habitat_mapper.target_list}")

        habitat_mapper.object_perceiver.sam.initialize(habitat_mapper.target)

        if args.relocate:
            habitat_agent.make_plan_mod_relocate(idx=i)
        else:
            habitat_agent.make_plan_mod_no_relocate(idx=i)

        flag = True
        while flag and not habitat_env.episode_over and habitat_agent.episode_steps < 500:
            flag = habitat_agent.step_mod(idx=i)

        habitat_agent.save_trajectory(f"./{args.save_dir}/episode-{i}/")
        evaluation_metrics.append({
            'Episode': i,
            'success': habitat_agent.metrics['success'],
            'spl': habitat_agent.metrics['spl'],
            'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
            'Episode Steps': habitat_agent.episode_steps,
            'start and goal distance': habitat_agent.start_end_episode_distance,
            'travel distance': habitat_agent.travel_distance,
            'Found Goal': habitat_agent.found_goal,
            'End': 1,
            'object_goal': habitat_agent.instruct_goal,
        })
        write_metrics(evaluation_metrics, path=f"./{args.save_dir}/metrics.csv")
        logger.info('\n')

        # except Exception as e:
        #     logger.exception(f"Error occurred in episode {i}: {e} \n")

        #     evaluation_metrics.append({
        #         'Episode': i,
        #         'success': habitat_agent.metrics['success'],
        #         'spl': habitat_agent.metrics['spl'],
        #         'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
        #         'Episode Steps': habitat_agent.episode_steps,
        #         'start and goal distance': habitat_agent.start_end_episode_distance,
        #         'travel distance': habitat_agent.travel_distance,
        #         'Found Goal': habitat_agent.found_goal,
        #         'End': 0,
        #         'object_goal': habitat_agent.instruct_goal,
        #     })
        #     write_metrics(evaluation_metrics, path=f"./{args.save_dir}/metrics.csv")
        #     logger.info('\n')
