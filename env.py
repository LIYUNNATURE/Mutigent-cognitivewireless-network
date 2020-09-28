import numpy as np
# import pandas as pd
import math
# import matplotlib.pyplot as plt
import argparse
import random

def transmission_packet(lambda_possion,k):
    A = np.random.poisson(lam=lambda_possion, size=k) 
    return A

def initial_user_state(n_agents,lifetime,lambda_possion):
    user_state = np.zeros((n_agents,lifetime),dtype=int)
    for i in range (n_agents):
        user_state[i][lifetime-1] =  (transmission_packet(lambda_possion,n_agents))[i]
    return user_state

def refresh_user_state(args,user_state,omega,lambda_possion):
    R = transmission_rate(omega)
    for user_number in range (args.n_agents):
        remain_packets = 0

        for n in range (args.lifetime-1):
            remain_packets = 0
            if(n>0):
                for j in range (1,n+1):
                    remain_packets = remain_packets + user_state[user_number][j]
            user_state[user_number][n] = max(((user_state)[user_number][n+1]-max((R[user_number]-remain_packets),0)),0)

        user_state[user_number][args.lifetime-1] = (transmission_packet(lambda_possion,args.n_agents))[user_number]
    return user_state

def transmission_rate(omega):
    
    C = [[1 for i in range(len(omega))]for j in range(len(omega))] 
    P = [100 for i in range(len(omega))] 
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

def get_user_loss(args,user_state):
    loss = [0 for i in range(args.n_agents)]
    for user_number in range(0,args.n_agents):
        loss[user_number] = user_state[user_number][0]
    loss_sum =  np.sum(loss,axis=0)
    return loss_sum

def get_user_payment(omega,args):
    user_payment = [0 for i in range(args.n_agents)]
    for user_number in range(args.n_agents):
        user_payment[user_number] = (transmission_rate(omega))[user_number]
    user_payment_sum = np.sum(user_payment,axis=0)
    return user_payment, user_payment_sum

def get_lambda_possion(step,lambda_possion):
    if step%2 == 0 : lambda_possion = [55,35]
    elif step%2 == 1 : lambda_possion = [35,40]
    else: lambda_possion = lambda_possion

    return lambda_possion

class Environment:
    def __init__(self,args):
        average_omega = args.omega_total/args.n_agents
        self.omega = [average_omega for i in range(args.n_agents)] #频谱分配
        self.user_state = [[0] * args.lifetime]*args.n_agents
        self.lambda_possion = args.lambda_possion
        self.n_agents = args.n_agents
        self.state = [0 for i in range(args.n_agents)]
        self.win_tag = False
        self.args =args
        self.reward = 0
        self.step = [0,0]
        self.obs=[]
        print('Initial Environment')

    def reset(self,args):
        average_omega = args.omega_total/args.n_agents
        # self.omega = [average_omega for i in range(args.n_agents)] #频谱分配
        self.omega = [44,26]
        self.user_state = [[0] * args.lifetime]*args.n_agents
        n_agents = self.n_agents
        lifetime = args.lifetime
        lambda_possion = self.lambda_possion

        self.user_state = initial_user_state(n_agents,lifetime,lambda_possion)
        error_counter = [0,0]
        
    def get_obs(self):
        omega = np.array(self.omega).reshape(2,1)
        state = np.array(self.state).reshape(2,1)
        step = np.array(self.step).reshape(2,1)
        obs = np.hstack((omega,state))
        obs = np.hstack((obs,step))
        self.obs = obs
        return obs

    def get_state(self):
        return self.state       

    def get_avail_agent_actions(self,agent_id,actions):
        # avail_agent_actions = [i for i in range(self.args.n_agents*self.args.n_agents)]
    #    average_omega = self.args.omega_total/args.n_agents
        if len(actions) == 0:
            if self.omega[agent_id] >= self.args.omega_total-6: avail_agent_actions = [1,2,3,0,0]
            elif self.omega[agent_id] >= self.args.omega_total-3: avail_agent_actions = [1,2,3,4,0]
            elif self.omega[agent_id] <= 3: avail_agent_actions = [0,0,3,4,5]
            elif self.omega[agent_id] <= 6: avail_agent_actions = [0,2,3,4,5]
            else: avail_agent_actions = [1,2,3,4,5]
        else:
            avail_agent_actions = [0,0,0,0,0]  
            if self.omega[agent_id]-6 > 0: avail_agent_actions[0] = 1
            if self.omega[agent_id]-3 > 0 and (sum(self.omega)+actions[0]-6-3)<self.args.omega_total: avail_agent_actions[1] = 2
            if (sum(self.omega)+actions[0])<self.args.omega_total:avail_agent_actions[2] = 3
            if (sum(self.omega)+actions[0]+3)<self.args.omega_total:avail_agent_actions[3] = 4
            if (sum(self.omega)+actions[0]+6)<self.args.omega_total:avail_agent_actions[4] =5

        return avail_agent_actions

    def forward_step(self, steps, action):
        # print('env-step-action:',action)
        # print('env-step-self.state:',self.state)
        last_state = self.state
        self.step = [steps%2,steps%2]
        args = self.args
        self.user_state = [[0] * args.lifetime]*args.n_agents
        n_agents = self.n_agents
        lifetime = args.lifetime
        
        self.lambda_possion = get_lambda_possion(steps, self.lambda_possion)
        error_counter = [0,0]
        reward = 0
        win_reward = 0
        temp=[0,0]
        all_gain = 0
        all_payment = 0
        all_loss = 0
        
        for i in range(args.n_agents):
            temp[i] = (action[i]-2)*3 + self.omega[i] 
        
        # while sum(temp) > 10:
        #      for i in range(args.n_agents):
        #         temp[i] -=1

        # while sum(temp) < 6:
        #     for i in range(args.n_agents):
        #         temp[i] +=1

        # while min(temp)<=0:  
        #     temp[temp.index(max(temp))]-=1
        #     temp[temp.index(min(temp))]+=1
        
        self.omega = temp 
        

        self.user_state = initial_user_state(n_agents,lifetime,self.lambda_possion)    

        for i in range(10):
            self.user_state = refresh_user_state(args,self.user_state, self.omega,self.lambda_possion)

        for i in range (args.communication_limit):
            loss = get_user_loss(args,self.user_state)
            gain, gain_sum= get_user_gain(args,self.user_state,self.omega)
            payment, payment_sum  = get_user_payment(self.omega,args)
            self.user_state = refresh_user_state(args,self.user_state, self.omega,self.lambda_possion)
            for i in range(args.n_agents):
                if self.user_state[i][0] != 0: 
                    error_counter[i] += 1
        # print('error_counter:',error_counter)
        for i in range(args.n_agents):
            if error_counter[i]==0: self.state[i] = 1
            else: self.state[i]=0

        if np.sum(error_counter) == 0: 
            self.win_tag = True
        else:
            self.win_tag = False

        win_tag = self.win_tag
        args.counter +=1

        if win_tag: win_reward = 100000
        else: win_reward = 0

        all_gain = sum(gain) + all_gain
        all_loss = loss + all_loss

        self.reward = reward 
        info = {"battle_won":win_tag,"action:":action,"self.omega:":self.omega,"self.state":self.state,"lambda-possion:":self.lambda_possion,"tramsmission_rate:":transmission_rate(self.omega),"step":self.step}
        # print('env-step-self.state-after_action:',self.state)
        return (all_gain,all_loss, win_reward, info)