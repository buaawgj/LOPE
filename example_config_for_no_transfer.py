############################################
# 修改 n_itr
# 修改 n_parallel
############################################

mode = "local"
train_low = True
train_high = True
train_low_with_penalty = True

n_parallel = 8 # 16
maze_id = 16 # 13

fence = False
n_itr = 200 # 1000 # number of iterations
death_reward = -0.0
sensor_range = 30.0
low_step_num = 2.5e5  # 6e5 for large maze, 5e4 for small maze
max_low_step = 1.2e4 # num of low steps in one episode, 3000 for large maze

train_high_every = 1
discount_high = 0.99 # changeable
success_reward = [500,]
exp_prefix_set = 'tmp'

animate = False # False
time_step_agg_anneal = False
anneal_base_number = 1.023293 # 100 * 1.023293 ** -100 = 10

direct_goal = False # whether to include goal as (x,y) or as 20 rays
random_start = False
velocity_field = False # whether to use manually-set velocity field as
train_low_with_v_split = False # use HAAR
train_low_with_v_gradient = False # useless
train_low_with_mmd_split = False
baseline_name = 'linear'
low_level_entropy_penalty = 0.

train_low_with_external = False # train with external rewards only, no auxiliary reward
itr_delay = 0

transfer = False

pkl_path = "/home/wgj/rllab/data/local/Swimmer-snn/egoSwimmer-snn_005MI_5grid_6latCat_bil_0012/params.pkl"

test_pkl_path = '/home/wgj/rllab/test/'

# 调试记录：

# 2021.01.21
# 将 train_low_with_v_split 改为 Flase;
# 将 train_low_with_mmd_split 改为 Flase;
# 2021.02.18 现在已经重新将上面两个参数设置为 True;

# 2021.02.18
# 在寻找多路径的环境中，需要对下面的参数进行重新设置
# random_start = False
# maze_id = 11
# max_low_step = 3000