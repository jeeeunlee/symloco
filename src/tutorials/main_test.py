import argparse
import os.path
import sys
import json

import scripts

log_path = "log"
data_path = "data"

if __name__ == "__main__":
    # example of python script for training and testing
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('-c', '--config', help="Path to config file", default="config/test.json")
    parser.add_argument('--rewardFile', help="Path to reward file", required=False)
    parser.add_argument('--logDir', help="Name of log directory to use for prediction", required=True)
    parser.add_argument('--step', help="Predict using the model after n time steps of training", required=False)

    # do not pass args to sub-functions for a better readability.
    args = parser.parse_args()

    # log path
    if args.rewardFile is not None:
        args.rewardFile = os.path.join("symlocogym", "rewards", args.rewardFile)

    # =============
    # prediction and saving
    # =============

    # managing paths and names
    if args.step is None:
        model_name = 'model'  # default to final model
    else:
        model_name = 'model_' + args.step + '_steps'
    load_path = os.path.join(log_path, args.logDir)  # example: name = 'log/2022-10-10-PylocoVanilla-v0-50.0M_3'
    
    # read config file
    with open(os.path.join(load_path, 'config.json'), 'r') as f:
        print("using config in the log folder...")
        params = json.load(f)
        env_id = params['env_id']

    max_predict_steps = params['train_hyp_params'].get('max_episode_steps_predict', 500)

    # prediction
    predict_env = scripts.predict_env_setup(env_id=env_id,
                                            load_path=load_path,
                                            env_params=params['environment_params'],
                                            reward_params=params['reward_params'],
                                            reward_path=args.rewardFile,
                                            max_predict_steps=max_predict_steps,
                                            verbose=1)

    scripts.predict(predict_env=predict_env, load_path=load_path, model_name=model_name,
                    max_predict_steps=max_predict_steps, render=True, verbose=1)
    # scripts.predict_with_command(predict_env, 2.0, load_path=load_path, model_name="model_20800000_steps",
    #                              verbose=0, name_suffix="2.0")
