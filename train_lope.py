import os
import sys
import gym
import time
import math 
import torch
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

from lope import LOPE
from utils import rollout, clip_dict, extend_array
from monitor import Monitor
import example_config_for_no_transfer as par
from rllab.envs.normalized_env import normalize
from sandbox.snn4hrl.envs.mujoco.maze.cheetah_maze_env import CheetahMazeEnv
from sandbox.snn4hrl.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv


def generate_data():
    data = np.load('hopper_trajs.npy', allow_pickle=True).tolist()
    clipped_data = []
    
    a = []
    for traj in data:
        traj['ret'] = 500
        traj['current_ep_reward'] = 500
        traj['goal'] = True
        # print("!!!!!!!!!coms: ", traj["coms"][-1]
        print("len: ", len(traj["rewards"]))
        print("return: ", np.sum(traj["rewards"][:500]))
        # print("x_position: ", traj["info"]["x_position"])
        indexes = np.argwhere(np.array(traj["coms"]) > 9.5)
        
        if len(indexes) == 0:
            pass
        elif len(indexes) != 0:
            index = indexes[0][0]
            clipped_traj = clip_dict(traj, index)
            clipped_traj = extend_array(clipped_traj, key="coms")
            clipped_data.append(deepcopy(clipped_traj))
            ret = sum(clipped_traj["rewards"])
            print("ret: ", ret)
            a.append(ret)
    
    print("a: ", a)    
    print("average return: ", np.mean(a))
    clipped_data = clipped_data[:5]
    
    return clipped_data

def train():
    ####### initialize environment hyperparameters ######
    num_policies = 1                      # the number of agents

    has_continuous_action_space = True    # continuous action space; else discrete

    max_ep_len = 500                      # max timesteps in one episode
    max_training_episodes = int(8e3)      # break training loop if timeteps > max_training_timesteps

    ################ PPO hyperparameters ################

    update_freq = 20                      # update policy every n timesteps
    K_epochs = 80                         # update policy for K epochs in one PPO update
    gamma = 0.99                          # discount factor
    random_seed = 99988                   # 56364  set random seed
    
    lr_actor = 0.0004                    # learning rate for actor network
    lr_critic = 0.001                     # learning rate for critic network

    use_smooth = True                       # set default value of the using_mmd
    alpha = 0.6                      # set the value of mmd_alpha
    min_alpha = 0.45                  # set the minimum of mmd_alpha
    alpha_decay_rate = 0.04           # set the decay rate of mmd_alpha
    alpha_decay_freq = 100            # set the decay frequency of mmd_alpha
    
    eps_clip = 0.27                       # clip parameter for PPO
    min_eps_clip = 0.11                   # set the minimum of eps_clip
    eps_clip_decay_rate = 0.0015          # set the decay rate of eps_clip 
    eps_clip_decay_freq = 200             # set the decay frequency of eps_clip
    
    action_std = 0.65                     # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.015        # linearly decay action_std
    min_action_std = 0.35                 # minimum action_std (stop decay after action_std <= min_action_std)
    min_action_std_1 = 0.15
    action_std_decay_freq = 125           # action_std decay frequency (in num timesteps)
    
    save_model_freq = 1000                # the frequency of saving model
    
    ################ Set Random Seed ##################
    
    if random_seed:
        print("--------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)

    ################ Create Environment ##################

    # env = normalize(SwimmerMazeEnv(
    #     maze_id=par.maze_id, death_reward=par.death_reward, 
    #     sensor_range=par.sensor_range, sensor_span=2*math.pi, 
    #     goal_rew=par.success_reward, random_start=par.random_start, 
    #     direct_goal=par.direct_goal, velocity_field=par.velocity_field))
    env_name = "Hopper-v4"
    env = gym.make(env_name)
    env = Monitor(env, max_ep_len, update_freq, use_sparse=True)
    
    state_dim = env.unwrapped.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.unwrapped.action_space.shape[0]
    else:
        action_dim = env.unwrapped.action_space.n
        
    known_trajs = []
    
    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "dense_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################
    
    for idx in range(num_policies):
        # create a ppo agent
        ppo_agent = LOPE(
            use_smooth, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, 
            eps_clip, has_continuous_action_space, alpha=alpha, action_std_init=action_std,
            known_trajs=known_trajs,
            )
        
        # initialize the writer
        env.init_results_writer(idx)
        
        # episode step for accumulate reward 
        epinfobuf = deque(maxlen=100)
        epinfos = deque(maxlen=update_freq)
        
        # Initialize parameters
        episode = 0
        ppo_agent.clear_trajs()
        
        # check learning time
        start_time = time.time()
        # record the step number
        step_num = 0
        # record the max_100_ep_rew
        results = []
        # record if the agent finds the goal
        find_goal = False 
        # record when the agents finds the goal 
        success_time = 0

        # training loop
        while episode <= max_training_episodes:
            # collect rollouts
            path = rollout(env, ppo_agent, max_ep_len)
            ppo_agent.append_traj(path)
            maybeepinfo = path['info'].get('episode')
            if maybeepinfo: 
                epinfos.append(maybeepinfo)
                epinfobuf.append(maybeepinfo)
            # record if the agent arrives at the goal
            if path["goal"] and not find_goal:
                find_goal = True
                success_time = episode
                
            # Record the number of steps.   
            step_num += len(path['observations'])
            episode += 1
            
            # if episode > 2999 and episode % 1000 == 0:
            #     ppo_agent.write_npy(episode)

            # update PPO agent and log data 
            if episode % update_freq == 0:
                paths = ppo_agent.return_trajs()
                loss, mmd_distances = ppo_agent.update(episode)
                ppo_agent.clear_trajs()
                
                # write data
                env.write_data(epinfos, mmd_distances)
                
                # check time interval
                time_interval = round(time.time() - start_time, 2)
                # calc mean return
                mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
                results.append(mean_100_ep_return)
                
                # Print log
                print('Used Step: ', step_num,
                    '| Loss: ', loss,
                    '| Mean ep 100 return: ', mean_100_ep_return,
                    '| Used Time:', time_interval)
                
            # save model weights
            if episode % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                # print("Elapsed Time  : ", float(datetime.now().replace(microsecond=0)) - start_time)
                print("--------------------------------------------------------------------------------------------")
             
            # adjust the value of eps_clip to stablize the training process
            if find_goal and episode >= 0.5 * max_training_episodes and episode % eps_clip_decay_freq == 0: 
                ppo_agent.decay_eps_clip(eps_clip_decay_rate, min_eps_clip)
                
            if episode >= 0.55 * max_training_episodes and episode % alpha_decay_freq == 0:
                ppo_agent.decay_alpha(alpha_decay_rate, min_alpha)
                    
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and episode >= 0.8 * max_training_episodes:
                if episode % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std_1)
                    
            elif has_continuous_action_space and episode >= 0.45 * max_training_episodes:
                if episode % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            

if __name__ == "__main__":
    for _ in range(1):
        train()
    
    # generate_data()