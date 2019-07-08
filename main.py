import tensorflow as tf
import numpy as np
from .parameters import Parameters as pa
from .env_define import *

MAX_EPISODE = 500
MAX_EPISODE_LEN = 1000

NUM_T = 1
NUM_R = 1
SIGMA2 = 0.3e-9

t_factor = 0.5
epsilon = 1.0

# config = {'state_dim': 3, 'action_dim': 2};
# train_config = {'minibatch_size': 64, 'actor_lr': 0.0001, 'tau': 0.001,
#                 'critic_lr': 0.001, 'gamma': 0.99, 'buffer_size': 1000000,
#                 'random_seed': 1234}
# user_config = [{'id': '1', 'model': 'AR', 'num_r': NUM_R, 'rate': 1.0, 'dis': 100, 'action_bound': 2,
#                 'data_buf_size': 100, 't_factor': t_factor, 'penalty': 1000, 'action_level': 5},
#                {'id': '2', 'model': 'AR', 'num_r': NUM_R, 'rate': 2.0, 'dis': 100, 'action_bound': 2,
#                 'data_buf_size': 100, 't_factor': t_factor, 'penalty': 1000, 'action_level': 5},
#                {'id': '3', 'model': 'AR', 'num_r': NUM_R, 'rate': 3.0, 'dis': 100, 'action_bound': 2,
#                 'data_buf_size': 100, 't_factor': t_factor, 'penalty': 1000, 'action_level': 5}]
user_config = [{'id': '1', 'model': 'AR', 'num_r': NUM_R, 'rate': 1.0, 'dis': 100, 'action_bound': 2,
                'data_buf_size': 100, 't_factor': t_factor, 'penalty': 1000, 'action_level': 5}]

# 0. initialize the session object
sess = tf.Session()

# 1. include all user in the system according to the user_config
user_list = [];
for info in user_config:
    user_list.append(SimEvnDQN(sess,parameters.Parameters()))
    print('test')

# 2. create the simulation env
env = Simenv(user_list,parameters.Parameters())

sess.run(tf.global_variables_initializer())

env.init_target_network(parameters.Parameters)

r_rec = []
# 3. start to explore for each episode
for i in range(MAX_EPISODE):

    # start from a random state
    cur_init_ds_ep = env.reset()
    cur_r_ep = np.zeros(len(user_list))
    cur_p_ep = np.zeros(len(user_list))
    cur_ts_ep = np.zeros(len(user_list))
    cur_ps_ep = np.zeros(len(user_list))
    cur_rs_ep = np.zeros(len(user_list))
    cur_ds_ep = np.zeros(len(user_list))
    cur_ch_ep = np.zeros(len(user_list))

    for j in range(MAX_EPISODE_LEN):

        # first try to transmit from current state
        [cur_r, done, cur_p, temp, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch] = env.step_transmit()

        cur_r_ep += cur_r
        cur_p_ep += cur_p
        cur_ts_ep += cur_ts
        cur_ps_ep += cur_ps
        cur_rs_ep += cur_rs
        cur_ds_ep += cur_ds
        cur_ch_ep += cur_ch

        if done:
            r_rec.append(cur_r)
            print('%d:r:%.4f,p:%.4f,tr:%.4f,pr:%.4f,rev:%.4f,dbuf:%.4f,ch:%.8f,ibuf:%d' % (
            i, cur_r_ep / MAX_EPISODE_LEN, cur_p_ep / MAX_EPISODE_LEN, cur_ts_ep / MAX_EPISODE_LEN,
            cur_ps_ep / MAX_EPISODE_LEN, cur_rs_ep / MAX_EPISODE_LEN, cur_ds_ep / MAX_EPISODE_LEN,
            cur_ch_ep / MAX_EPISODE_LEN, cur_init_ds_ep[0]))

# plot_helper(ran