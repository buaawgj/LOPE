import os
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from process_traj import distance_to_buffer
from prefer_memory import PreferBuf


################################## set device ##################################

print("===============================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("===============================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        action_std_init, 
        has_continuous_action_space, 
        ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh())
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1))
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))
        
        # critic for smooth rewards 
        self.smooth_critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        smooth_state_values = self.smooth_critic(state)
        
        return action_logprobs, state_values, smooth_state_values, dist_entropy


class LOPE(nn.Module):
    RECORD = "trajs.npy"
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    
    def __init__(
        self, 
        use_smooth,
        state_dim, 
        action_dim, 
        lr_actor, 
        lr_critic, 
        gamma, 
        K_epochs, 
        eps_clip, 
        has_continuous_action_space, 
        action_std_init=0.65,
        batch_size=256,
        alpha=0.55,
        max_num=5,
        goal_num=1,
        maxlen=40,
        epsilon=1e-5,
        known_trajs=None,
        k=2,
        ):
        super(LOPE, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.use_smooth = use_smooth
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip 
        
        self.batch_size = batch_size
        self.buffer = RolloutBuffer()
        self.terminated = False
        self.alpha = alpha
        
        if self.use_smooth: 
            self.lr_actor = lr_actor / 2
        else:
            self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.policy = ActorCritic(
            state_dim, action_dim, action_std_init, has_continuous_action_space
            ).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
            {'params': self.policy.smooth_critic.parameters(), 'lr': self.lr_critic},
            ])

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std_init, has_continuous_action_space
            ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.policies = []
        self.trajs = []
        self.preferred_memory = PreferBuf(goal_num)
        self.max_num = max_num
        self.k = k
        self.known_trajs = known_trajs
        self.epsilon = epsilon
        self.file_path = os.path.join(self.__class__.PROJECT_ROOT, self.__class__.RECORD)
        
    def clear_trajs(self):
        del self.trajs[:]   
    
    def append_traj(self, traj):
        self.trajs.append(traj)
    
    def return_trajs(self):
        return self.trajs
    
    def write_npy(self, episode):
        record = "trajs_{}.npy".format(episode)
        file_path = os.path.join(self.__class__.PROJECT_ROOT, record)
        trajs = np.array(self.trajs)
        np.save(file_path, trajs)
    
    def decay_eps_clip(self, eps_clip_decay_rate, min_eps_clip):
        print("-----------------------------------------------------------------------------------")
        self.eps_clip = self.eps_clip - eps_clip_decay_rate 
        if self.eps_clip <= min_eps_clip:
            self.eps_clip = min_eps_clip 
            print("setting eps_clip to min_eps_clip : ", min_eps_clip)
    
        else:
            print("setting eps_clip to : ", self.eps_clip)    
        print("-----------------------------------------------------------------------------------")
        
    def decay_alpha(self, alpha_decay_rate, min_alpha):
        print("-----------------------------------------------------------------------------------")
        self.alpha = self.alpha - alpha_decay_rate 
        if self.alpha <= min_alpha:
            self.alpha = min_alpha
            print("setting alpha to min_eps_clip : ", min_alpha)
        
        else:
            print("setting alpha to : ", self.alpha)    
        print("-----------------------------------------------------------------------------------")

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("-------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("-------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("-----------------------------------------------------------------------------------")
        
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("-----------------------------------------------------------------------------------")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()
        
    def find_worst_traj(self):
        max_length = 0
        max_id = None
        for idx, traj in enumerate(self.known_trajs):
            length = len(traj["observations"])
            if length > max_length:
                max_id = idx 
                max_length = length 
        
        return max_id, max_length
    
    def update_known_trajectories(self):
        for traj in self.trajs:
            if traj['goal']:
                if len(self.known_trajs) >= self.max_num:
                    max_id, max_length = self.find_worst_traj() 
                    print("length of demo traj: ", max_length)
                    assert max_id != None, "max_id must not be None!"
                    
                    print("length of traj: ", len(traj["observations"]))
                    if len(traj["observations"]) < max_length: 
                        self.known_trajs.pop(max_id)
                        self.known_trajs.append(deepcopy(traj))
                        print("Successfully update the demonstration memory!") 
                        
                elif len(self.known_trajs) < self.max_num:
                    self.known_trajs.append(deepcopy(traj))
                    print("Successfully add a trajectory to the replay memory!")
    
    def distance_to_demos(self, path, key='coms', h=3, gap=3):
        expert_buffer = self.preferred_memory.trajectory_memory
        
        temp_list = deque(maxlen=1)
        max_x = 0
        max_idx = None
        for i, expert_trajectory in enumerate(expert_buffer):
            if expert_trajectory['coms'][-1][0] > max_x:
                max_x = expert_trajectory['coms'][-1][0]
                max_idx = i
                temp_list.append(deepcopy(expert_trajectory))
                
        expert_features = []
        for expert_traj in temp_list:
            expert_features.append(expert_traj[key]) 
        
        current_trajectory = path[key]
        mmd_distance, idx = distance_to_buffer(current_trajectory, expert_features, h=h, gap=gap)
        print("x_position: ", temp_list[0]["coms"][-1])
        return mmd_distance, idx 
    
    def compute_smooth_guidance(self, alpha=0.5):
        """
        params: trajs -> the trajectories generated in the current epoch, we hope each trajectory containing the return information of the past good trajectory closest to it and the MMD distance information.
        params: alpha -> the coefficient for computing joint rewards.
        return: trajs -> we add some new attributions to each trajectory in buffer trajs.
        """
        beta = 1 - alpha
        dists = []
        all_contributions = []
        for traj in self.trajs:
            dists.append(traj['dist'])
            # TGPO requirement
            all_contributions.append(1 - traj['dist'])
        
        # self.valbuf.add_values(dists)
        
        # dists = self.valbuf.obtain_values()
        # omega_denominator = sum(np.exp(-self.k * np.array(dists))) 
        
        # all_contributions = []
        # for traj in self.trajs:
        #     # compute the value of omega 
        #     omega_numerator = np.exp(-self.k * traj["dist"])
        #     omega = omega_numerator / (omega_denominator + self.epsilon) 
        #     print("Omega: ", omega)
            
        #     joint_reward = alpha * traj["ret"] + beta * traj["past_good_return"]
        #     contribution = omega * joint_reward 
        #     print("Contribution: ", contribution)
            
        #     all_contributions.append(contribution)
        
        max_contribution = max(all_contributions)
        min_contribution = min(all_contributions)
        delta = max_contribution - min_contribution
        print("delta: ", delta)
        
        # TGPO requirement
        all_contributions = np.array(all_contributions)
        all_contributions = (all_contributions - all_contributions.mean()) / (all_contributions.std() + 1e-7)
        
        for idx, contribution in enumerate(all_contributions):
            traj = self.trajs[idx]
            # normalized_contri = (contribution - min_contribution) / delta
            normalized_contri = contribution
            smooth_rewards = normalized_contri * np.ones((len(traj["rewards"]),))
            print("normalized contribution: ", normalized_contri)
                
            self.trajs[idx]["smooth_rewards"] = smooth_rewards
            
        return self.trajs

    def update(self, episode):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        
        # Evaluating old actions and values
        (
            logprobs, 
            old_state_values, 
            old_smooth_state_values, 
            dist_entropy  
        ) = self.policy.evaluate(old_states, old_actions)
        
        # match state_values tensor dimensions with rewards tensor
        old_state_values = torch.squeeze(old_state_values)
        old_smooth_state_values = torch.squeeze(old_smooth_state_values)
        # Finding Surrogate Loss
        advantages = rewards - old_state_values.detach()
        
        ################ calculate smooth rewards ################
        mmd_distances = []
        known_trajectories = self.preferred_memory.trajectory_memory
        if self.use_smooth and known_trajectories:
            if episode < 1000:
                h = 2.5
            elif episode >= 1000 and episode < 4000:
                h = 8.5
            elif episode >= 4000:
                h = 9.5
            for idx, path in enumerate(self.trajs):
                mmd_distance, expert_id = self.distance_to_demos(path, key='coms', h=h, gap=6)
                mmd_distances.append(mmd_distance)
                print("!!!!!!!!mmd_distance: ", mmd_distance)
                
                path["past_good_return"] = known_trajectories[expert_id]["ret"]
                path["dist"] = mmd_distance 
            
            self.trajs = self.compute_smooth_guidance()
            
            all_smooth_rewards = []
            for traj in self.trajs: 
                all_smooth_rewards += traj["smooth_rewards"].tolist()
                
            discount_smooth_returns = []
            discount_smooth_return = 0 
            for smooth_reward, is_terminal in zip(reversed(all_smooth_rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discount_smooth_return = 0 
                    
                discount_smooth_return = smooth_reward + (self.gamma * discount_smooth_return)
                discount_smooth_returns.insert(0, discount_smooth_return)
                
            smooth_returns = torch.tensor(discount_smooth_returns, dtype=torch.float32).to(device)
            smooth_returns = (smooth_returns - smooth_returns.mean()) / (smooth_returns.std() + 1e-7)
            
            # compute the smooth advantages
            smooth_advantages = smooth_returns - old_smooth_state_values.detach()
            
            # # Adaptive sigma scaling method
            # if meet_same_goal:
            #     self.mmd_alpha = self.mmd_alpha * 1.5
            # elif not meet_same_goal and self.mmd_alpha > 1.20:
            #     self.mmd_alpha = self.mmd_alpha * 0.8
            # elif not meet_same_goal and self.mmd_alpha > 0.20:
            #     self.mmd_alpha = self.mmd_alpha * 0.96
            # elif not meet_same_goal and self.mmd_alpha > 0.15 and episode > 240:
            #     self.mmd_alpha = self.mmd_alpha * 0.99
            # print("!!!!!!!!!!!mmd_alpha: ", self.mmd_alpha)
        
        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            # Evaluating old actions and values
            (
                logprobs, 
                state_values, 
                smooth_state_values, 
                dist_entropy  
            ) = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            smooth_state_values = torch.squeeze(smooth_state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # advantages = rewards - state_values.detach()  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # the original PPO objective
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            ################ calculate the smooth loss ################
            if self.use_smooth and self.preferred_memory.trajectory_memory:
                # compute smooth reward advantages
                # smooth_advantages = smooth_returns - smooth_state_values.detach()
                smooth_surr1 = ratios * smooth_advantages 
                smooth_surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * smooth_advantages 
                smooth_loss = -torch.min(smooth_surr1, smooth_surr2)
            
            # final loss of clipped objective PPO
            if not self.use_smooth or not self.preferred_memory.trajectory_memory:
                loss = loss.mean()
            elif self.use_smooth and self.preferred_memory.trajectory_memory:
                loss = loss.mean() \
                       + self.alpha * smooth_loss.mean() \
                       + 0.5 * self.MseLoss(smooth_state_values, smooth_returns) \
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # # update the past good trajectory buffer
        # if self.use_smooth:
        #     self.update_known_trajectories() 
        
        # update the preferred memory
        self.preferred_memory.update(self.trajs)
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        print("action_std: ", self.action_std)
        print("eps_clip: ", self.eps_clip)
        print("alpha: ", self.alpha)
        print("episode: ", episode)
        
        return loss, mmd_distances
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))