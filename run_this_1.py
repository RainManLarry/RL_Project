from RL_brain import DeepQNetwork
from env_define import *
from parameters import Parameters

def run_simEnv(config):
    step = 0

    # reward_list=np.zeros(config.max_episode*config.slot_num)
    # step_list=np.zeros(config.max_episode*config.slot_num,int)
    # count=0

    for episode in range(config.max_episode):
        # initial observationconfig,step_num act as rq_id
        env = Simenv(pa)
        observation = env.reset()
        req_id = 0
        slot_id=0
        for j in range(config.slot_num):

            reward_sum=0
            for i in range(0,config.slot_request_num-1,config.request_handled_num):

                # if np.random.uniform() < pa.epsilon:
                action = RL.choose_action(observation)
                #
                # else :
                #     action =env.choose_action_random(observation)
                # action = env.choose_action_random(observation,config)
                step_num=j
                # RL take action and get next observation and reward

                observation_, reward = env.step(action,req_id,slot_id,config)
                print(reward)
                RL.store_transition(observation, action, reward, observation_)

                if (step > 200) and (step % 5 == 0):
                    RL.learn()

                # swap observation
                observation = observation_

                step += 1
                req_id += 1
                reward_sum +=reward
                # print(step)
            slot_id +=1
            env.updat_slot_status(slot_id,pa.slot_request_num)
            # reward_list[episode*config.slot_num+slot_id]=reward_sum
            # step_list[episode*config.slot_num+slot_id]=episode*config.slot_num+slot_id






if __name__ == "__main__":

    pa= Parameters()
    env = Simenv(pa)
    RL = DeepQNetwork(env.action_dim_num, env.state_dim_num,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=1000,
                      # output_graph=True
                      )
    run_simEnv(pa)
    RL.plot_cost()
