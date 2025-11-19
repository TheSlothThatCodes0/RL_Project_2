# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert recorded demonstration actions between IK and joint space."""

import argparse
import multiprocessing
import os
from copy import deepcopy

import torch
from tqdm import tqdm

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

def convert_joint_to_ik(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF state + gripper)."""
    try:
        eef_state = ep_data.data["obs"]["ee_frame_state"]
        joint_actions = ep_data.data["actions"]

        gripper_action = joint_actions[:, -1:]
        new_actions = torch.cat([eef_state, gripper_action], dim=1)

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK: {str(e)}")

def convert_ik_to_joint(ep_data: EpisodeData) -> EpisodeData:
    """Convert IK actions to joint targets."""
    try:
        joint_targets = ep_data.data["obs"]["joint_pos_target"]
        ep_data.data["actions"] = joint_targets
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert IK to joint: {str(e)}")

ACTION_CONVERTERS = {
    "ik": convert_joint_to_ik,
    "joint": convert_ik_to_joint,
}

def process_dataset(input_file: str, output_file: str, action_type: str) -> None:
    """Process dataset episodes and convert actions to the desired type."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset file does not exist: {input_file}")

    converter = ACTION_CONVERTERS[action_type]

    input_handler = HDF5DatasetFileHandler()
    output_handler = HDF5DatasetFileHandler()

    input_handler.open(input_file)
    output_handler.create(output_file)

    try:
        episode_names = list(input_handler.get_episode_names())
        skipped_episodes = []
        
        for name in tqdm(episode_names, desc="Processing episodes"):
            try:
                ep_data = input_handler.load_episode(name, device="cpu")

                if ep_data.success is not None and not ep_data.success:
                    continue

                processed = deepcopy(ep_data)
                processed = converter(processed)
                output_handler.write_episode(processed)
                
            except Exception as e:
                skipped_episodes.append((name, str(e)))
                print(f"\nWarning: Skipping episode '{name}' due to error: {str(e)}")
                continue
        
        if skipped_episodes:
            print(f"\n\nSummary: Skipped {len(skipped_episodes)} episode(s) due to errors:")
            for ep_name, error_msg in skipped_episodes:
                print(f"  - {ep_name}: {error_msg}")

    finally:
        input_handler.close()
        output_handler.flush()
        output_handler.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert recorded demonstration actions between IK and joint space."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./datasets/annotated_dataset.hdf5",
        help="Path to input dataset file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/processed_annotated_dataset.hdf5",
        help="Path to save processed dataset file."
    )
    parser.add_argument(
        "--action_type",
        choices=["ik", "joint"],
        required=True,
        help="Target action representation: 'ik' or 'joint'."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    process_dataset(args.input_file, args.output_file, args.action_type)

if __name__ == "__main__":
    main()
