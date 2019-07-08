import numpy as np
import math




class Parameters:
    def __init__(self):

        self.cloudlet_filename = 'config/cloudlet.csv'
        self.network_filename='config/topology.csv'
        self.vnf_filename='config/vnf.csv'
        self.request_filename = 'config/request.csv'

        self.max_episode = 1
        self.max_episode_len = 1200
        # 1200 request 5  *240  or 10*120  or 15*80 or 20*60  mean slot 'num
        self.slot_num=5
        self.slot_request_num=240

        # self.max_episode = 1
        # self.max_episode_len = 200
        # self.slot_num=1
        # self.slot_request_num=200

       # input config
        self.cloudlet_num=20  #cloudlet node num
        self.vnf_num=10
        self.request_handled_num=1

      # state_dim=cloud_num( cloudlet_num*(cloudlet_cap_rem+vnf_num*
      # ( instance_num，accumulate packet，min_delay)
      # + schedule_request_num( locknodeid, packet rate ,delay ,duration)
      #   =state_dim=cloudlet_num*(cloudlet(1+vnf_num*4)
        self.state_dim=self.cloudlet_num*(1+self.vnf_num*3)+self.request_handled_num*5

        # action_dim =request_handled_num*(rqid,handle_cloudlet_nodeid,action_type)
        # action_type 0 reject 1 create 2 used old
        # self.action_dim=self.request_handled_num*3
        # self.action_dim=self.request_handled_num*self.cloudlet_num*3
        self.action_dim = self.request_handled_num * self.cloudlet_num * 2+1
        # user_config
        self.id='1'
        self.rate=3
        self.dis= 100
        self.action_bound=1
        self.data_buf_size=100
        self.t_factor=1.0
        self.penalty= -2000

        # reward set

        self.reward_admin=1
        self.reward_reject = 0
        self.reward_violate=-1
        # instance vnf  time low and high
        self.delay_ins_min=20
        self.delay_ins_max=50
        # each cloudlet cost per MHZ in each slot
        self.cprice_cal_slot=0.25
        # is the half of network node
        self.max_step_len=25


        #train_fig

        self.minibatch_size=64
        self.actor_lr=0.0001
        self.tau=0.001
        self.critic_lr=0.001
        self.gamma=0.99
        self.buffer_size=1000000
        self.random_seed=1234
        self.epsilon=0.9






        # self.num_epochs = 10000         # number of training epochs
        # self.simu_len = 10             # length of the busy cycle that repeats itself
        # self.num_ex = 1                # number of sequences
        #
        # self.output_freq = 10          # interval for output and store parameters







