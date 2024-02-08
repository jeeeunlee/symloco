import argparse
import os.path
import sys
import json

import scripts

log_path = "log"
data_path = "data"

if __name__ == "__main__":
    # example of python script for training
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('-c', '--config', help="Path to config file", default="config/test.json")
    parser.add_argument('-g', '--group', help="Name of group", required=False)
    parser.add_argument('-d', '--debug', help="Debug mode.", required=False, action='store_true')
    parser.add_argument('-vr', '--videoRecorder', help="Activate video recorder to record a video footage.",
                        required=False, action='store_true')
    parser.add_argument('--rewardFile', help="Path to reward file", required=False)
    parser.add_argument('-wb', '--wandb', help="Enable logging to wandb", required=False, action='store_true')

    # do not pass args to sub-functions for a better readability.
    args = parser.parse_args()

    # log path
    if args.rewardFile is not None:
        args.rewardFile = os.path.join("symlocogym", "rewards", args.rewardFile)


    # =============
    # training
    # =============

    # config file
    print('- config file path = {}'.format(args.config))
    with open(args.config, 'r') as f:            
        params = json.load(f)

    # training
    scripts.train(
        params=params,
        log_path=log_path,
        debug=args.debug,
        video_recorder=args.videoRecorder,
        wandb_log=args.wandb,
        group_name=args.group,
        config_path=args.config,
        reward_path=args.rewardFile
    )