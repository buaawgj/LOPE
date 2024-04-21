import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def extend_array(traj, key):
    coms = []
    for x in traj[key]:
        extend_x = np.zeros((2,))
        extend_x[0] = x 
        coms.append(extend_x)
        
    traj[key] = coms
    
    return traj

def clip_dict(traj, index):
    """
    A function clips trajectories.
    """ 
    ret = {}
    keys = list(traj.keys())
    
    for key in keys:
        example = traj[key]
        
        if isinstance(example, dict):
            v = clip_dict(example, index)
        elif isinstance(example, list):
            v = example[:index+1]
        else:
            v = example  
             
        ret[key] = v
        
    return ret

def select_action(policy, o, epsilon=0, flag=False):
    o = torch.from_numpy(np.array(o)).float().unsqueeze(0)
    probs = policy(o)
    dist = Categorical(probs)
    action = dist.sample()
        
    if flag:
        print("action : ", action)
        print("action prob : ", probs[0, action[0]])
            
    return action.item(), dist.log_prob(action)

def rollout(env, agent, max_path_length=np.inf, test=False):
    observations = []
    actions = []
    rewards = []
    log_probs = []
    is_terminals = []
    coms = []
    goal = None
    ret = 0

    o, _ = env.reset()
    
    for t in range(max_path_length): 
        a = agent.select_action(o)                
        next_o, reward, done, info = env.step(a)
        
        ret += reward
        
        if not test:
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
        elif test:
            agent.terminated = done
        
        observations.append(o)
        rewards.append(reward)
        actions.append(a)
        coms.append(np.array([info["x_position"], 0])) 
        # log_probs.append(log_prob)
        
        if t == max_path_length - 1:
            print("com: ", info['x_position'])
    
        if done:
            print("com: ", info['x_position'])
            if info['x_position'] > 10:
                goal = True
            break
        
        o = next_o
    
    path = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        is_terminals=is_terminals,
        log_probs=log_probs,
        coms=coms,
        goal=goal,
        info=info,
        ret=ret,
        )
    
    return path