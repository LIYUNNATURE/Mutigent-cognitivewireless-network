import  math
import numpy as np
from arguments import get_common_args, get_mixer_args
import matplotlib.pyplot as plt

def transmission_packet(lambda_possion,k):
    A = np.random.poisson(lam=lambda_possion, size=k) 
    return A

def initial_user_state(n_agents,lifetime,lambda_possion):
    user_state = np.zeros((n_agents,lifetime),dtype=int)
    for i in range (n_agents):
        user_state[i][lifetime-1] =  (transmission_packet(lambda_possion,n_agents))[i]
    return user_state

def refresh_user_state(args,user_state,omega):
    R = transmission_rate(omega)

    for user_number in range (args.n_agents):
        remain_packets = 0

        for n in range (args.lifetime-1):
            remain_packets = 0
            if(n>0):
                for j in range (1,n+1):
                    remain_packets = remain_packets + user_state[user_number][j]
            user_state[user_number][n] = max(((user_state)[user_number][n+1]-max((R[user_number]-remain_packets),0)),0)

        user_state[user_number][args.lifetime-1] = (transmission_packet(args.lambda_possion,args.n_agents))[user_number]
    return user_state

def transmission_rate(omega):
    
    C = [[1 for i in range(len(omega))]for j in range(len(omega))] 
    P = [100000 for i in range(len(omega))] 
    N = [1 for i in range(len(omega))] 
    R = [0 for i in range(len(omega))]
    for i in range(len(omega)):
        if omega[i] == 0: R[i] = 0
        else: R[i] = round(omega[i]*math.log(1+(C[i][i]*P[i]/N[i]/omega[i])))
    return R

def get_user_gain(args,user_state,omega):
    gain_array = [0 for i in range(args.n_agents)]
    gain_sum = 0
    for user_number in range(0,args.n_agents):
        gain = 0
        R = transmission_rate(omega)
        for n in range (args.lifetime-1):
            remain_packets = 0
            if(n>=1):
                for j in range (1,n+1):
                    remain_packets = remain_packets + user_state[user_number][j]
            gain = max(0,min(user_state[user_number][n+1],(R[user_number]-remain_packets)))+gain
        gain_array[user_number] = gain 
        gain_sum = np.sum(gain_array,axis=0)
    return gain_array, gain_sum

def get_user_payment(omega,args):
    user_payment = [0 for i in range(args.n_agents)]
    for user_number in range(args.n_agents):
        user_payment[user_number] = (transmission_rate(omega))[user_number]
    user_payment_sum = np.sum(user_payment,axis=0)
    return user_payment, user_payment_sum

def _get_max_episode_len(terminated):
    episode_num = terminated.shape[0]
    max_episode_len = 0
    print('agent-episode_num',episode_num)
    for episode_idx in range(episode_num):
        for transition_idx in range(100):
            if terminated[episode_idx, transition_idx, 0] == 1:
                if transition_idx + 1 >= max_episode_len:
                    max_episode_len = transition_idx + 1
                break
    print('agent-max_episode_len:',max_episode_len)
    return max_episode_len

def step(args, action,state):
    user_state = [[0] * args.lifetime]*args.n_agents
    n_agents = args.n_agents
    lifetime = args.lifetime
    lambda_possion = args.lambda_possion
    error_counter = [0,0,0,0]
    reward = 0
    win_number = 0
    # manner = [-2,-2,-2,-2]

    # if np.array(self.state).any()>=0 or np.array(np.array(self.state)+np.array(manner)).any()>0:
    #     self.state = np.array(action)+np.array(self.state)+np.array(manner)

    for i in range(args.n_agents):

        temp = action[i] + state[i] -2
        if temp>0: state[i] = temp
        else: state[i] = 0

    user_state = initial_user_state(n_agents,lifetime,lambda_possion)    

    for i in range(10):
        user_state = refresh_user_state(args,user_state,state)

    for i in range (args.communication_limit):
        user_state = refresh_user_state(args,user_state, state)
        gain, gain_sum= get_user_gain(args,user_state,state)
        payment, payment_sum  = get_user_payment(state,args)
        for i in range(args.n_agents):
            if user_state[i][0] != 0: 
                error_counter[i] += 1

    if np.sum(error_counter) == 0: 
        win_tag = True
    else:
        win_tag = False

    terminated = False
    if args.counter == 500 or win_tag == True :
        terminated = True
        args.counter = 0
    args.counter +=1
    if win_tag: reward +=  gain_sum - payment_sum
    else: reward += gain_sum-payment_sum
    info = {"battle_won":win_tag,"self.state":state,"reward":reward}
    # print('env-step-self.state-after_action:',self.state)
    return (reward, terminated, info)

omega = np.arange(1,10)
rate = transmission_rate(omega)

plt.plot(omega,rate)
plt.show()
# avail_actions = []
# avail_actions_ind = np.nonzero(avail_actions)[0] 
# avail_actions.append([1,2,3])
# print(avail_actions)