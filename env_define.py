import numpy as np
import networkx as nx
from RL_brain import DeepQNetwork as RL
from parameters import Parameters
import matplotlib.pyplot as plt

# request representation
class request:
    def __init__(self, rq_id, rq_loc_id, rq_sf_id,rq_packrate, rq_delay,rq_duration):
        self.rq_id = rq_id
        self.rq_sf_id = rq_sf_id
        self.rq_loc_id = rq_loc_id
        self.rq_packrate=rq_packrate
        self.rq_delay=rq_delay
        self.rq_duration=rq_duration
        # -1 init 0 reject 1 doing 2 done
        self.rq_status=-1
        self.rq_start_time=0
        self.rq_cloudlet_num=-1
        self.rq_admit_type=-1
        self.rq_end_time=0

    # each slot schedule  request
    def admit(self,cloudlet_num,admit_type,start_time):
        if(admit_type!=0):
           self.rq_status=1
           self.rq_cloudlet_num = cloudlet_num
           self.rq_start_time = start_time
           self.rq_end_time = start_time + self.rq_duration -1
        else:
            self.rq_status=0



    # each slot handle adminted request,judge weather request is finish ,need to change system state
    def handle(self):
        # -1 init 0 reject 1 doing 2 done
        self.rq_status=2
        return self.rq_cloudlet_num, self.rq_sf_id, self.rq_packrate


# service function representation
class Servicefunction:
    def __init__(self, sf_id, sf_max_rate, sf_rc_unit):
        self.sf_id = sf_id
        self.sf_max_rate = sf_max_rate
        self.sf_rc_unit = sf_rc_unit
        self.sf_rc_alloc=sf_rc_unit*sf_max_rate

# cloudlet representation
class Cloudlet:
    def __init__(self,cl_id,switch_id,cl_cap_cp,vnf_num):

        self.cl_id=cl_id
        self.switch_id=switch_id
        self.cl_cap_cp=cl_cap_cp
        self.cl_cap_used=0
        self.cl_cap_rem=cl_cap_cp
        self.cl_sf_num_list=np.zeros(vnf_num)  # servicefunction instance num
        self.cl_sf_sum_rate_list=np.zeros(vnf_num) #servicefunction handle request packrate sum
        self.cl_sf_rq_num_list=np.zeros(vnf_num) #servicefunction handle request num



    # cloudlet install  service function instance
    def ins_servicefun(self,sf_id,sf_rc_alloc):
        if(self.cl_cap_rem>sf_rc_alloc):
            self.cl_cap_used=self.cl_cap_used+ sf_rc_alloc
            self.cl_sf_num_list[sf_id]=self.cl_sf_num_list[sf_id]+1
            self.cl_cap_rem=self.cl_cap_rem - sf_rc_alloc
            return True
        else:
            return False

    # cloudlet release  service function instance
    def rel_servicefun(self,sf_id,sf_rc_alloc):
        if(self.cl_sf_num_list[sf_id]>1):
            self.cl_cap_used=self.cl_cap_used- sf_rc_alloc*self.cl_sf_num_list[sf_id]
            self.cl_sf_num_list[sf_id] = 0
            self.cl_cap_rem=self.cl_cap_rem+sf_rc_alloc*self.cl_sf_num_list[sf_id]


        # cloudlet release  service function instance

    def admit_request(self,sf_id,sf_rc_alloc,rq_packrate,action_type):
        # create new instance
        if(action_type==1 or self.cl_sf_rq_num_list[sf_id]==0):
            self.ins_servicefun(sf_id,sf_rc_alloc)

        self.cl_sf_sum_rate_list[sf_id] +=rq_packrate
        self.cl_sf_rq_num_list[sf_id] += 1

    # return release vnf instance flag
    def finish_request_cloudlet(self,sf_id,prate):
        self.cl_sf_sum_rate_list[sf_id] -= prate
        self.cl_sf_rq_num_list[sf_id] -= 1

        if(self.cl_sf_rq_num_list[sf_id]==0):

           return  True



class Simenv:
    def __init__(self,pa):
        self.G = nx.Graph()  # G is network

        self.cloudlet_list = []# cloudlet liest
        self.vnf_list=[] #service function list
        self.request_list=[]# request list



        self.state_dim_num= pa.state_dim
        self.action_dim_num=pa.action_dim
        self.vnf_num=pa.vnf_num

        self.cloulet_num=pa.cloudlet_num
        self.request_handle_num=pa.request_handled_num

        self.current_state=np.zeros(self.state_dim_num)

        # init network
        netfile = open(pa.network_filename)
        i = 0
        for line in netfile:
            if i >= 0:
                col = line.split(',')
                self.G.add_weighted_edges_from([(col[0],col[1],float(col[2]))])
            i += 1
        netfile.close()


        self.node_list=list(self.G.nodes())
        self.delay_path = dict(nx.all_pairs_shortest_path(self.G))
        self.delay_length= dict(nx.all_pairs_dijkstra_path_length(self.G))

        # self.edge_list=self.G.edges()

        # init clouldlet_list

        cloudletfile = open(pa.cloudlet_filename)
        j = 0
        for lines in cloudletfile:
            if j >= 0:
                cols = lines.split(',')
                cloudlet=Cloudlet(cols[0],cols[0],float(cols[1]),pa.vnf_num)
                self.cloudlet_list.append(cloudlet)
            j += 1
        cloudletfile.close()

       # init vnf_list

        vnffile = open(pa.vnf_filename)
        k = 0
        for linef in vnffile:
            if k >= 0:
                colf = linef.split(',')
                vnf=Servicefunction(colf[0],float(colf[1]),float(colf[2]))
                self.vnf_list.append(vnf)
            k += 1
        vnffile.close()

        # init req_list

        reqfile = open(pa.request_filename)
        k = 0
        for liner in reqfile:
            if k >= 0:
                colf = liner.split(',')
                req = request(int(colf[0]), colf[1], colf[2],int(colf[3]),int(colf[4]),int(colf[5]))
                self.request_list.append(req)
            k += 1
        reqfile.close()

    def sel_by_name(self,name,type):
        if(type=='v'):
            for x in self.vnf_list:
                if x.sf_id == name:
                    return self.vnf_list.index(x)
        elif (type == 'c'):
            for x in self.cloudlet_list:
                if x.cl_id == name:
                    return self.cloudlet_list.index(x)
        elif (type == 'n'):
            return self.node_list.index(name)
        # elif (type == 'r'):
        #     for x in self.request_list:
        #         if str(x.rq_id) == name:
        #             return self.cloudlet_list.index(x)
        else:
            return  -1
    # cal each admit request reward
    def cal_reward(self,rq_id,cloudlet_id,action_type,pa):

        delay_net=0
        delay_handle=0
        delay_que=0
        delay_ins=0
        delay=0
        if(action_type==0):
            return pa.reward_reject
        else:
           source_node=self.request_list[rq_id].rq_loc_id
           des_node=self.cloudlet_list[cloudlet_id].cl_id
           delay_net=self.sel_delay_time(source_node,des_node)
           if(delay_net!=-1):
               delay_net=2*delay_net
           if(action_type==1):
               # delay_ins=np.random.randint(pa.delay_ins_min,pa.delay_ins_max)
               delay_ins = np.random.randint(20, 50)

           vnf_index=self.sel_by_name(self.request_list[rq_id].rq_sf_id,'v')
           each_handle_rate=self.vnf_list[vnf_index].sf_max_rate
           delay_handle=1/each_handle_rate

           ins_num=self.cloudlet_list[cloudlet_id].cl_sf_num_list[vnf_index]
           sum_packet=self.cloudlet_list[cloudlet_id].cl_sf_sum_rate_list[vnf_index]

           if (action_type == 2):
               delay_que=1/(each_handle_rate*ins_num-sum_packet)


           delay=delay_handle+delay_ins+delay_net+delay_que

           if(delay>self.request_list[rq_id].rq_delay):
               return  pa.reward_violate
           else:
               cost=0
               cost_ins=0
               cost_handle=0
               cost_trans=0

               # instance vary from 20-50
               if(action_type==1):
                   cost_ins=20+(vnf_index%7)*5

               prate=self.request_list[rq_id].rq_packrate
               rduration=self.request_list[rq_id].rq_duration

               cost_handle=prate*rduration*pa.cprice_cal_slot
               cost_handle_max=100*5*pa.cprice_cal_slot
               # delay_net = transfer in each slot per packet

               cost_trans=prate*rduration*delay_net/1000
               cost_trans_max=prate*rduration*pa.max_step_len/1000

               cost=(cost_handle + cost_trans)/(cost_handle_max + cost_trans_max)

               return  pa.reward_admin-cost

    # setp_num is request id  slot id
    def step(self, action,step_num,slot_id,pa):
        ob_state=np.zeros(self.state_dim_num,dtype=np.int)
        reward=np.zeros(self.request_handle_num)
        # get action context
        for k in range(self.request_handle_num):
            # rq_id=int(action[k * self.request_handle_num])
            # cloudlet_id=int(action[k * self.request_handle_num + 1])
            # act_type=int(action[k * self.request_handle_num + 2])

            # revise because rq_id not in state
            rq_id=step_num
            # cloudlet_id=int(action[k * self.request_handle_num])
            # act_type=int(action[k * self.request_handle_num +1])

            # action  return  a value  act_type =0,1,2 reject ,new ,used old
            #---this  act method one
            # act=divmod(action,3)
            # cloudlet_id= act[0]
            # act_type= act[1]

            # ---this  act method one

            if(action==len(self.cloudlet_list)*2):
                act_type=0
                cloudlet_id=0
            else:
                act=divmod(action,2)
                cloudlet_id= act[0]
                act_type= act[1]+1






            # print(action)
            # print("cloudlet_id")
            # print(act[0])
            # print("act_type")
            # print(act[1])
            # update_request_list_state(rq_id, act_type)
            self.request_list[rq_id].admit(cloudlet_id, act_type, slot_id)

            if(act_type>0):
                 # updte cloudlet_list_state
                 sf_id=self.sel_by_name(self.request_list[rq_id].rq_sf_id,'v')
                 sf_rc_alloc=self.vnf_list[sf_id].sf_rc_alloc
                 rq_packrate=self.request_list[rq_id].rq_packrate
                 self.cloudlet_list[cloudlet_id].admit_request(sf_id,sf_rc_alloc,rq_packrate,act_type)

            # cal action reward
            reward[k]=self.cal_reward(rq_id,cloudlet_id,act_type,pa)


        # set current_state --cloudlet state
        # state_dim=cloud_num( cloudlet_num*(cloudlet_cap_rem+vnf_num*
        # ( instance_num，accumulate packet，min_delay)
        # + schedule_request_num( locknodeid, packet rate ,delay ,duration)
        #   =state_dim=cloudlet_num*(cloudlet(1+vnf_num*4)
        for i in range (len(self.cloudlet_list)):
           ob_state[i*(pa.vnf_num*3+1)]=self.cloudlet_list[i].cl_cap_rem
           for j in range(pa.vnf_num):
             ob_state[i * (pa.vnf_num * 3+1 )+j*3 + 1]=self.cloudlet_list[i].cl_sf_num_list[j]
             ob_state[i * (pa.vnf_num * 3 +1)+j*3 + 2]= self.cloudlet_list[i].cl_sf_sum_rate_list[j]
             ob_state[i * (pa.vnf_num * 3 +1)+j*3 + 3] = 0

        k=0
        rq_id +=1
        for k in range(self.request_handle_num):
          if(rq_id+k<=len(self.request_list)-1):
             # ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*6-5] =rq_id + k
             ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*5-4]=self.sel_by_name( self.request_list[rq_id + k].rq_loc_id, 'n')
             ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*5-3] = self.sel_by_name(self.request_list[rq_id + k].rq_sf_id, 'v')
             ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*5-2] = self.request_list[rq_id + k].rq_packrate
             ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*5-1] = self.request_list[rq_id + k].rq_delay
             ob_state[self.state_dim_num-1-(self.request_handle_num-k-1)*5-0] = self.request_list[rq_id + k].rq_duration
          else:
              # ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 6 - 5] = -1
              ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 5 - 4] = -1
              ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 5 - 3] = 0
              ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 5 - 2] = 0
              ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 5 - 1] = 0
              ob_state[self.state_dim_num - 1 - (self.request_handle_num - k - 1) * 5 - 0] = 0

        self.current_state=ob_state



        return ob_state ,reward


    def reset(self,isTrans=True):
        self.current_state=np.zeros(self.state_dim_num)
        for i in range(len(self.cloudlet_list)):
            self.current_state[i * (self.vnf_num * 3 + 1)] = self.cloudlet_list[i].cl_cap_rem
            for j in range(self.vnf_num):
                self.current_state[i * (self.vnf_num * 3 + 1) + j * 3 + 1] = self.cloudlet_list[i].cl_sf_num_list[j]
                self.current_state[i * (self.vnf_num * 3 + 1) + j * 3 + 2] = self.cloudlet_list[i].cl_sf_sum_rate_list[j]
                self.current_state[i * (self.vnf_num * 3 + 1) + j * 3 + 3] = 0
        for k in range(self.request_handle_num):
           # self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*6 - 5]=self.request_list[k].rq_id
           self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*5-4] = self.sel_by_name(self.request_list[k].rq_loc_id,'n')
           self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*5- 3] = self.sel_by_name(self.request_list[k].rq_sf_id,'v')
           self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*5- 2] = self.request_list[k].rq_packrate
           self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*5- 1] = self.request_list[k].rq_delay
           self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*5- 0] = self.request_list[k].rq_duration

        return self.current_state

    def sel_delay_time(self,source,des):
        if source in self.delay_length and des in self.delay_length[source]:
            return self.delay_length[source][des]
        else:
            return -1

    def updat_slot_status(self,slot_num,slot_request_num):
        #  request duration between 1-5
        # if(slot_num>5):
        #     start_id=(slot_num-5+1)*slot_request_num
        #     end_id=(slot_num+1)*slot_request_num
        # else:
        #     start_id=0
        #     end_id=(slot_num+1)*slot_request_num

        for i in range(0,slot_request_num):
            # get request id that need to handle next round
            end_time= self.request_list[i].rq_end_time
            status=self.request_list[i].rq_status
            if(status==1 and end_time==slot_num):
                cloudlet_id, rq_sf_id, rq_packrate=self.request_list[i].handle()
                sf_id=self.sel_by_name(rq_sf_id,"v")
                del_flag=self.cloudlet_list[cloudlet_id].finish_request_cloudlet(sf_id,rq_packrate)
                # if have done request and no request use than rel_vnf_instance
                if(del_flag):
                    self.cloudlet_list[cloudlet_id].rel_servicefun(sf_id,self.vnf_list[sf_id].sf_rc_alloc)







        return True
    # judge path is exist,second if  has used same type than reuse ,else  if has enough space then alloc new
    # request_state (locknodeid, packet rate, delay, duration)
    def choose_action_random(self,observation,pa):
        action=np.zeros(pa.action_dim)
        for k in range(self.request_handle_num):
            rq_id=int(self.current_state[self.state_dim_num - 1 -(self.request_handle_num-k-1)*6 - 5])
            rq_loc_name=self.request_list[rq_id].rq_loc_id
            rq_vnf_id=int(self.sel_by_name(self.request_list[rq_id].rq_sf_id,'v'))
            vnf_cap_req=self.vnf_list[rq_vnf_id].sf_rc_alloc
            flag=True
            step=1
            while flag:
                cloudlet_id=np.random.randint(0,pa.cloudlet_num)
                den_cloudlet_name=self.cloudlet_list[cloudlet_id].cl_id
                # judge there is a path between rq_loca_name & cloudlet_name
                distance=self.sel_delay_time(rq_loc_name,den_cloudlet_name)
                if(distance>=0):

                    action[k * self.request_handle_num] = rq_id
                    action[k * self.request_handle_num + 1] = cloudlet_id
                    if (self.cloudlet_list[cloudlet_id].cl_sf_num_list[rq_vnf_id]>0 and self.cloudlet_list[cloudlet_id].cl_cap_rem>vnf_cap_req):
                        action[k * self.request_handle_num + 2] = np.random.randint(1, 3)
                        flag = False
                    elif (self.cloudlet_list[cloudlet_id].cl_sf_num_list[rq_vnf_id]>0 ):
                        action[k * self.request_handle_num + 2] = 2
                        flag=False
                    elif(self.cloudlet_list[cloudlet_id].cl_cap_rem>self.vnf_list[rq_vnf_id].sf_rc_alloc):
                        action[k * self.request_handle_num + 2] = 1
                        flag = False
                    else:
                        flag=True

                else:
                    if(step<20):
                        step += 1
                    else:
                        flag=False
                        # reject this request
                        action[k * self.request_handle_num] = rq_id
                        action[k * self.request_handle_num + 1]=-1
                        action[k * self.request_handle_num + 2] = 0

        return action

# class SimEvnDQN(Simenv):
#     def __init__(self, sess, pa):
#         Simenv.__init__(self, pa)
#         self.sess = sess
#         self.action_level = pa.action_level
#         # self.agent = DQNAgent(sess, pa)
#         self.action = 0
#
#         # self.table = np.array(
#         #     [[float(self.action_bound) / (self.action_level - 1) * i for i in range(self.action_level)] for j in
#         #      range(self.action_dim)])
# #  unit test
def run_main(pa,env):

   step = 0
   for episode in range(pa.episode_max_length):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            if np.random.uniform() < self.epsilon:
            # RL choose action based on observation
                action = RL.choose_action(observation)
            else:
                action=env.choose_action_random(pa)


            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

             # swap observation
            observation = observation_
        if done:
            break
        step += 1


   print('game over')

if __name__ == '__main__':
    print('game over')
    # pa = Parameters()
    # env = Simenv(pa)
    # # run_main(pa,env)
    # # print(env.sel_by_name('6','r'))
    #
    # step = 0
    # for episode in range(pa.max_episode):
    #     # initial observation
    #     observation = env.reset()
    #     # print(observation)
    #
    #     for j in range(pa.slot_num):
    #
    #         for i in range(0,pa.slot_request_num, pa.request_handled_num):
    #
    #
    #             action = env.choose_action_random(observation,pa)
    #             print('action -------')
    #             print(action)
    #             step_num = j
    #             # RL take action and get next observation and reward
    #             observation_ = env.step(action, step_num)
    #
    #             print('current_state -------')
    #             print(observation_)
    #
    #
    #
    #
    #
    # print('game over ')



