import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, win_number, episode_num=None, evaluate=False):
        # if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
        #     self.env.close()
        
        win_number = 0
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        all_gain,all_loss = [],[]
        self.env.reset(self.args)
        # terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        info = {"battle_won":False}

        # epsilon
        epsilon = 1 if not evaluate else 0
        
        while step < self.episode_limit: # and not info["battle_won"]:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            if step%2 == 0:
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id,actions)
                    action = self.agents.choose_action(obs[agent_id], state[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot
            else:
                for agent_id in range(self.n_agents-1,-1,-1):
                    avail_action = self.env.get_avail_agent_actions(agent_id,actions)
                    action = self.agents.choose_action(obs[agent_id], state[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate)
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.insert(0,action)
                    actions_onehot.insert(0,action_onehot)
                    avail_actions.insert(0,avail_action)
                    last_action[agent_id] = action_onehot


            gain, loss, reward, info = self.env.forward_step(step,actions) #采取行为之后的omega的奖励以及信息

            if info['battle_won']: win_number += 1 
            all_gain.append(gain)
            all_loss.append(loss)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            # terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if self.args.epsilon_anneal_scale == 'step':
            #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            # print("o is {},s is {}".format(o,s))
            # if evaluate: print('rollout-info:',info)
            print('rollout-info:',info)

        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:] #list切片除去第一个值
        s_next = s[1:] 
        o = o[:-1] #list切片除去最后一个值
        s = s[:-1]
        
        # print('o:',o)
        # print('o_next:',o_next)

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id,actions)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            # terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                    #    terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate: #evaluate == True: epsilon更新
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, info , win_number,all_gain,all_loss