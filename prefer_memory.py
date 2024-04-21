import numpy as np 
from copy import deepcopy 

from utils import clip_dict


class PreferBuf():
    """ 
    Collect preference-labelled trajectories.
    
    params: goal_num: bool -> it means if multiple goals exist in the environment
    params: threshold: float -> 
    """
    def __init__(
        self, 
        goal_range=1,
        goal_num=1, 
        threshold=1.,
        prior_traj_num=5, 
        maze_size_scaling=3,
        ):
        self.goal_range = goal_range
        self.goal_num = goal_num 
        self.threshold = threshold
        self.prior_traj_num = prior_traj_num
        self.memory_size = goal_num * prior_traj_num
        
        if self.goal_num > 1:
            self.prior_memory = {}
        else: 
            self.prior_memory = [] 
        
        self.preferred_trajs = []
        self.num_preferred_trajs = 8
        self.do_prefer = True
        self.maze_size_scaling = maze_size_scaling
        
        # preferred_targets depends on the environment.
        # You need to set a 2-dimension numpy array.
        # self.preferred_targets = np.array([[6, 0], [6, 5.5], [5, 5.5], [0, 6]])
        # self.check_points = np.array([[4, 0], [6, 1], [6, 3.5], [6, 5.5]])
        # self.preferred_targets = np.array([[2, 0], [3, 0], [4, 0], [5, 0]])
        # self.check_points = np.array([[1.5, 0], [2.5, 0], [3.5, 0], [4.5, 0]])
        
        self.preferred_targets = np.array([[1.5, 0], [5, 0], [8, 0], [10, 0]])
        self.check_points = np.array([[0.5, 0], [1.5, 0], [5, 0], [8, 0]])
    
    @property
    def trajectory_memory(self):
        if self.goal_num == 1 and self.do_prefer:
            return self.prior_memory + self.preferred_trajs 
        elif self.goal_num == 1 and not self.do_prefer:
            return self.prior_memory 
        else:
            raise NotImplementedError
            
    def update(self, paths, stored_goals=None, i_actor=None):
        """
        Update the trajectory replay memory.
        """
        # if self.goal_num == 1:
        #     below_max_length = len(self.prior_memory) < self.prior_traj_num
        #     # store the trajectories to the goals in the prior memory
        #     self.update_single_memory(paths)
            
        # elif self.goal_num > 1:
        #     # determine the value of below_max_length
        #     if "treasure" in self.prior_memory.keys():
        #         below_max_length = len(self.prior_memory["treasure"]) < self.prior_traj_num
        #     elif "treasure" not in self.prior_memory.keys():
        #         below_max_length = True
                
        #     # store the trajectories to the goal in the prior memory
        #     self.update_multiple_memory(paths)
        
        # # update the preferred memory
        # if below_max_length:
        #     self.update_preferred_memory(paths, stored_goals, i_actor) 
        # elif not below_max_length:
        #     self.do_prefer = False
        self.update_preferred_memory(paths, stored_goals, i_actor) 
                
    def update_preferred_memory(self, paths, stored_goals=None, i_actor=None):
        """
        Store preferred trajectories in the preferred set.
        The goals, "apple" and "treasure", are hardcoded. 
        """
        if self.goal_num > 1 and "apple" in stored_goals:
            if stored_goals[i_actor] == None:
                for idx, path in enumerate(paths): 
                    self.determine_preferred_traj(path)
                        
            elif stored_goals[i_actor] == "treasure":
                has_treasure = "treasure" not in self.prior_memory.keys()
                below_max_length = len(self.prior_memory["treasure"]) < self.prior_traj_num
                if has_treasure and below_max_length:
                    for idx, path in enumerate(paths):
                        self.determine_preferred_traj(path)
        
        elif self.goal_num == 1:
            for idx, path in enumerate(paths):
                self.determine_preferred_traj(path)
            
        else: pass
                
    def update_multiple_memory(self, paths):
        """
        The prior memory stores trajectories to goals.
        Update the prior memory when there only exist multiple goals in a task.
        """
        num_for_each_category = dict()
        for goal in self.prior_memory.keys():
            num_for_each_category[goal] = len(self.prior_memory[goal])
            
        for idx, path in enumerate(paths):
            if path["goal"] == None: 
                pass
            
            elif path["goal"] not in num_for_each_category.keys():
                self.prior_memory[path["goal"]] = [deepcopy(path)]
                
            elif num_for_each_category[path["goal"]] < self.prior_traj_num:
                self.prior_memory[path["goal"]].append(deepcopy(path))
                
            elif num_for_each_category[path["goal"]] == self.prior_traj_num:
                min_reward, max_length, replace_idx = self.preferred_threshold(path["goal"])
                
                if path["current_ep_reward"] >= min_reward and len(path["observations"]) <= max_length:
                    pop_term = self.prior_memory[path["goal"]].pop(replace_idx)
                    # Check! 
                    if pop_term["current_ep_reward"] != min_reward:
                        print("ERROR!!!")
                        break
                    
                    self.prior_memory[path["goal"]].append(deepcopy(path))       
                    print("There is a path to be replaced!")
                    
                else:
                    print("The length of current path: ", len(path["observations"]))
                    print("The minimum return of good paths: ", min_reward)
                    print("There is no path to be replaced!") 
                    
    def update_single_memory(self, paths):
        """
        The prior memory stores trajectories to goals.
        Update the prior memory when there only exists a single goal in a task.
        """
        for idx, path in enumerate(paths):
            is_preferred = path['goal']
            
            if not is_preferred: pass
            
            elif is_preferred and len(self.prior_memory) < self.memory_size:
                self.prior_memory.append(deepcopy(path))
                print("Congratulation: There is a path to be added!")

            elif is_preferred and len(self.prior_memory) >= self.memory_size:
                min_reward, max_length, replace_idx = self.preferred_threshold()
                if path["current_ep_reward"] >= min_reward and len(path["observations"]) <= max_length:
                    self.prior_memory.pop(replace_idx)
                    self.prior_memory.append(deepcopy(path))
                    print("Congratulation: There is a path to be replaced!")
                else: 
                    print("Pity: There is no path to be replaced!")
    
    def judge(self, path): 
        """
        To determine whether the path is a good path or not.
        We need define different standards for different environments.
        """
        ret = path['rewards'][-1]
        sz = len(path['observations'])
        if ret < self.threshold:
            return False
        
        replace_id = None
        for idx, good_path in enumerate(self.prior_memory):
            good_ret = good_path['returns'][-1] 
            good_sz = len(good_path['observations'])
            if ret >= good_ret and sz < good_sz:
                return True 
            return False 

    def preferred_threshold(self, goal=None): 
        """
        To determine whether the path is a better path or not.
        We need define different standards for different environments.
        """
        if goal == None:
            memory = self.prior_memory 
        elif goal != None:
            memory = self.prior_memory[goal]  
        
        replace_idx = None
        min_reward = 1e4
        max_length = -1e3 
                      
        for idx, good_path in enumerate(memory):
            ret = good_path["current_ep_reward"]
            sz = len(good_path["observations"])

            if ret <= min_reward and sz >= max_length:
                min_reward = ret 
                max_length = sz
                replace_idx = idx
        
        return min_reward, max_length, replace_idx
    
    def check_buffer(self, goal):
        if len(self.prior_memory[goal]) >= self.prior_traj_num:
            return True 
        return False
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def determine_preferred_traj(self, init_path):
        # self.preferred_targets = np.array([[1, 35], [1, 24], [12, 15]]) 
        flag = None
        index_0 = None 
        index_1 = None 
        index_2 = None 
        index_3 = None
        
        tmp_list = []
        for idx, observation in enumerate(init_path["coms"]):
            tmp_list.append(observation)
        
        # determine if the trajectory enter the top-middle room
        a = self.check_points[3][0]
        b = self.check_points[3][1]
        boundary = self.maze_size_scaling * 0.5
        
        # list_3 = list(filter(lambda m: m[0] >= a and m[1] >= -boundary, init_path["coms"]))
        # if list_3: 
        #     min_dist_3 = 1e4
        #     min_idx_3 = None 
        #     for idx, state in enumerate(list_3):
        #         dist = np.linalg.norm(np.array(state[:2]) - self.preferred_targets[3])
        #         if dist < min_dist_3:
        #             min_dist_3 = dist
        #             min_idx_3 = idx 
        #     if min_idx_3 != None:
        #         index_3 = np.array(tmp_list).tolist().index(np.array(list_3)[min_idx_3].tolist())
        #     flag = "Fourth"
            
        x_positions = np.array(init_path["coms"])[:, 0]
        idx_cross_checkpoint_3 = np.where(x_positions >= a)[0]
        
        if idx_cross_checkpoint_3.size > 0:
            min_dist_3 = 1e4
            min_idx_3 = None 
            for idx in idx_cross_checkpoint_3:
                dist = np.linalg.norm(np.array(init_path['coms'][idx][:2]) - self.preferred_targets[3])
                if dist < min_dist_3:
                    min_dist_3 = dist
                    min_idx_3 = idx 
                    
            if min_idx_3 != None:
                index_3 = min_idx_3
            flag = "Fourth"
            
        idx_cross_checkpoint_2 = np.array([])
        if flag == None:
            # determine if the trajectory enter the top-right room
            a = self.check_points[2][0]
            b = self.check_points[2][1]
            
            x_positions = np.array(init_path["coms"])[:, 0]
            idx_cross_checkpoint_2 = np.where(x_positions >= a)[0]
            
            # list_2 = list(filter(lambda m: m[0] >= a and m[1] >= b - boundary, init_path["coms"]))
            
        if idx_cross_checkpoint_2.size > 0:
            min_dist_2 = 1e4
            min_idx_2 = None 
            for idx in idx_cross_checkpoint_2:
                dist = np.linalg.norm(np.array(init_path['coms'][idx][:2]) - self.preferred_targets[2])
                if dist < min_dist_2:
                    min_dist_2 = dist
                    min_idx_2 = idx 
                    
            if min_idx_2 != None:
                index_2 = min_idx_2
            flag = "Third"
            
        # if list_2:
        #     min_dist_2 = 1e4
        #     min_idx_2 = None 
        #     for idx, state in enumerate(list_2):
        #         dist = np.linalg.norm(np.array(state[:2]) - self.preferred_targets[2])
        #         if dist < min_dist_2:
        #             min_dist_2 = dist
        #             min_idx_2 = idx 
        #     if min_idx_2 != None:
        #         index_2 = np.array(tmp_list).tolist().index(np.array(list_2)[min_idx_2].tolist())
        #     flag = "Third"
        
        # Avoid redundant computation.
        idx_cross_checkpoint_1 = np.array([])
        if flag == None:
            # determine if the trajectory enter the top-right room
            a = self.check_points[1][0]
            b = self.check_points[1][1]
            
            # list_1 = list(filter(lambda m: m[0] >= a and m[1] >= b - boundary, init_path["coms"]))
            x_positions = np.array(init_path["coms"])[:, 0]
            idx_cross_checkpoint_1 = np.where(x_positions >= a)[0]
            
        if idx_cross_checkpoint_1.size > 0:
            min_dist_1 = 1e4
            min_idx_1 = None 
            for idx in idx_cross_checkpoint_1:
                dist = np.linalg.norm(np.array(init_path['coms'][idx][:2]) - self.preferred_targets[1])
                if dist < min_dist_1:
                    min_dist_1 = dist
                    min_idx_1 = idx 
                    
            if min_idx_1 != None:
                index_1 = min_idx_1
            flag = "Second"
            
        # if list_1:
        #     min_dist_1 = 1e4
        #     min_idx_1 = None 
        #     for idx, state in enumerate(list_1):
        #         dist = np.linalg.norm(np.array(state[:2]) - self.preferred_targets[1])
        #         if dist < min_dist_1:
        #             min_dist_1 = dist
        #             min_idx_1 = idx 
        #     if min_idx_1 != None:
        #         index_1 = np.array(tmp_list).tolist().index(np.array(list_1)[min_idx_1].tolist())
        #     flag = "Second"
        
        # Avoid redundant computation. 
        idx_cross_checkpoint_0 = np.array([])
        if flag == None: 
            # determine if the trajectory enter the bottom-right room
            a = self.check_points[0][0]
            b = self.check_points[0][1]
            # list_0 = list(filter(lambda m: m[0] >= a and m[1] >= -boundary, init_path["coms"]))
            x_positions = np.array(init_path["coms"])[:, 0]
            idx_cross_checkpoint_0 = np.where(x_positions >= a)[0]
            
        if idx_cross_checkpoint_0.size > 0:
            min_dist_0 = 1e4
            min_idx_0 = None 
            for idx in idx_cross_checkpoint_0:
                dist = np.linalg.norm(np.array(init_path['coms'][idx][:2]) - self.preferred_targets[0])
                if dist < min_dist_0:
                    min_dist_0 = dist
                    min_idx_0 = idx 
                    
            if min_idx_0 != None:
                index_0 = min_idx_0
            flag = "First"
        
        # if list_0:
        #     min_dist_0 = 1e4
        #     min_idx_0 = None 
        #     for idx, state in enumerate(list_0):
        #         dist = np.linalg.norm(np.array(state[:2]) - self.preferred_targets[0])
        #         if dist < min_dist_0:
        #             min_dist_0 = dist 
        #             min_idx_0 = idx 
        #     if min_idx_0 != None:
        #         index_0 = np.array(tmp_list).tolist().index(np.array(list_0)[min_idx_0].tolist())
        #     flag = "First"
        
        print("flag: ", flag)
        
        # Obtain segments of trajectories. 
        path = {}
        if flag == "First" and index_0 != None:
            path = deepcopy(clip_dict(init_path, index_0))
            path["flag"] = flag
        
        elif flag == "Second" and index_1 != None:
            path = deepcopy(clip_dict(init_path, index_1))
            path["flag"] = flag
            
        elif flag == "Third" and index_2 != None:
            path = deepcopy(clip_dict(init_path, index_2))
            path["flag"] = flag
            
        elif flag == "Fourth" and index_3 != None:
            path = deepcopy(clip_dict(init_path, index_3))
            path["flag"] = flag
        
        # Compute distances to different pre-defined targets.
        if path != {}:
            triple_end = np.tile(path["coms"][-1][:2], (4, 1))
            distance = np.linalg.norm(triple_end - self.preferred_targets, axis=1)
            path["distance"] = distance 
        
        # Store the preferred trajectory.
        if len(self.preferred_trajs) < self.num_preferred_trajs and path != {}: 
            self.preferred_trajs.append(deepcopy(path))
            print("path: ", path["coms"])
        
        # Substitude a preferred trajectory in the set.
        elif len(self.preferred_trajs) >= self.num_preferred_trajs and path != {}:
            if path["flag"] == "First": 
                max_dist = 0
                replace_idx = None
                for idx, preferred_traj in enumerate(self.preferred_trajs):
                    if preferred_traj["flag"] == "First" and preferred_traj["distance"][0] > max_dist:
                        max_dist = preferred_traj["distance"][0]
                        replace_idx = idx 
                    
                if path["distance"][0] < max_dist:
                    self.preferred_trajs.pop(replace_idx)
                    self.preferred_trajs.append(deepcopy(path))
                    print("First replace First!!!")
                    print("path: ", path["coms"])

            elif path["flag"] == "Second": 
                max_dist_0 = 0 
                replace_idx_0 = None 
                max_dist_1 = 0 
                replace_idx_1 = None
                replace_type = "Second"
                
                for idx, preferred_traj in enumerate(self.preferred_trajs):
                    if "flag" in preferred_traj.keys() and preferred_traj["flag"] == "First":
                        if replace_type == "Second":
                            replace_type = "First"
                            if preferred_traj["distance"][0] > max_dist_0:
                                max_dist_0 = preferred_traj["distance"][0]
                                replace_idx_0 = idx 
                
                    if preferred_traj["flag"] == "Second":
                        if replace_type == "Second":
                            if preferred_traj["distance"][1] > max_dist_1:
                                max_dist_1 = preferred_traj["distance"][1]
                                replace_idx_1 = idx         
                        elif replace_type == "First":
                            pass 
                    
                if replace_type == "First" and replace_idx_0 != None:
                    self.preferred_trajs.pop(replace_idx_0)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Second replace First!!!")
                    print("path: ", path["coms"])
                        
                elif replace_type == "Second" and replace_idx_1 != None:
                    if path["distance"][1] < max_dist_1:
                        self.preferred_trajs.pop(replace_idx_1)
                        self.preferred_trajs.append(deepcopy(path))
                        print("Second replace Second!!!")
                        print("path: ", path["coms"])

            elif path["flag"] == "Third": 
                max_dist_0 = 0  
                replace_idx_0 = None 
                max_dist_1 = 0 
                replace_idx_1 = None
                max_dist_2 = 0
                replace_idx_2 = None 
                replace_type = "Third"
                
                for idx, preferred_traj in enumerate(self.preferred_trajs):
                    if preferred_traj["flag"] == "First":
                        if replace_type != "First":
                            replace_type = "First"
                            if preferred_traj["distance"][0] > max_dist_0:
                                max_dist_0 = preferred_traj["distance"][0]
                                replace_idx_0 = idx 
                            
                    if preferred_traj["flag"] == "Second":
                        if replace_type == "Third":
                            replace_type = "Second"
                            max_dist_1 = preferred_traj["distance"][1]
                            replace_idx_1 = idx 
                        elif replace_type == "Second":
                            if preferred_traj["distance"][1] > max_dist_1:
                                max_dist_1 = preferred_traj["distance"][1]
                                replace_idx_1 = idx      
                        elif replace_type == "First":
                            pass 
                        
                    if preferred_traj["flag"] == "Third":
                        if replace_type == "First" or replace_type == "Second":
                            pass 
                        elif replace_type == "Third":
                            if preferred_traj["distance"][2] > max_dist_2:
                                max_dist_2 = preferred_traj["distance"][2]
                                replace_idx_2 = idx
                        
                if replace_type == "First" and replace_idx_0 != None:
                    self.preferred_trajs.pop(replace_idx_0)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Third replace First!!!")
                    print("path: ", path["coms"])
                        
                elif replace_type == "Second" and replace_idx_1 != None:
                    self.preferred_trajs.pop(replace_idx_1)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Third replace Second!!!")
                    print("path: ", path["coms"])
                    
                elif replace_type == "Third" and replace_idx_2 != None:
                    print("x_replaced: ", self.preferred_trajs[replace_idx_2]["coms"][-1])
                    print("x_current: ", path["coms"][-1])
                    if path["distance"][2] < max_dist_2:
                        self.preferred_trajs.pop(replace_idx_2)
                        self.preferred_trajs.append(deepcopy(path))
                        print("Third replace Third!!!")
                        print("path: ", path["coms"])
                        
            elif path["flag"] == "Fourth": 
                max_dist_0 = 0  
                replace_idx_0 = None 
                max_dist_1 = 0 
                replace_idx_1 = None
                max_dist_2 = 0
                replace_idx_2 = None 
                max_dist_3 = 0
                replace_idx_3 = None
                replace_type = "Fourth"
                
                for idx, preferred_traj in enumerate(self.preferred_trajs):
                    if preferred_traj["flag"] == "First":
                        if replace_type != "First":
                            replace_type = "First"
                            max_dist_0 = preferred_traj["distance"][0]
                            replace_idx_0 = idx 
                        elif replace_type == "First":
                            if preferred_traj["distance"][0] > max_dist_0:
                                max_dist_0 = preferred_traj["distance"][0]
                                replace_idx_0 = idx 
                            
                    elif preferred_traj["flag"] == "Second":
                        if replace_type == "Fourth" or replace_type == "Third":
                            replace_type = "Second"
                            max_dist_1 = preferred_traj["distance"][1]
                            replace_idx_1 = idx 
                        elif replace_type == "Second":
                            if preferred_traj["distance"][1] > max_dist_1:
                                max_dist_1 = preferred_traj["distance"][1]
                                replace_idx_1 = idx      
                     
                    elif preferred_traj["flag"] == "Third":
                        if replace_type == "Fourth":
                            replace_type = "Third"
                            max_dist_2 = preferred_traj["distance"][2]
                            replace_idx_2 = idx
                        elif replace_type == "Third":
                            if preferred_traj["distance"][2] > max_dist_2:
                                max_dist_2 = preferred_traj["distance"][2]
                                replace_idx_2 = idx
                                
                    elif preferred_traj["flag"] == "Fourth":
                        if replace_type == "Fourth":
                            if preferred_traj["distance"][3] > max_dist_3:
                                max_dist_3 = preferred_traj["distance"][3]
                                replace_idx_3 = idx
                                
                print("replace_idx_0: ", replace_idx_0)
                print("replace_idx_1: ", replace_idx_1)
                print("replace_idx_2: ", replace_idx_2)
                print("replace_idx_3: ", replace_idx_3)
                        
                if replace_type == "First" and replace_idx_0 != None:
                    self.preferred_trajs.pop(replace_idx_0)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Fourth replace First!!!")
                    print("path: ", path["coms"])
                        
                elif replace_type == "Second" and replace_idx_1 != None:
                    self.preferred_trajs.pop(replace_idx_1)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Fourth replace Second!!!")
                    print("path: ", path["coms"])
                    
                elif replace_type == "Third" and replace_idx_2 != None:
                    self.preferred_trajs.pop(replace_idx_2)
                    self.preferred_trajs.append(deepcopy(path))
                    print("Fourth replace Third!!!")
                    print("path: ", path["coms"])
                        
                elif replace_type == "Fourth" and replace_idx_3 != None:
                    if path["distance"][3] < max_dist_3:
                        self.preferred_trajs.pop(replace_idx_3)
                        self.preferred_trajs.append(deepcopy(path))
                        print("Fourth replace Fourth!!!")
                        print("path: ", path["coms"])
                        
    # def obstain_lowest_priority_traj(self):
    #     for idx, path in self.
    #     pass