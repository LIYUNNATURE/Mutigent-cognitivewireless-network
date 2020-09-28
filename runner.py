from agent import Agents
from rollout import RolloutWorker
from replay_buffer import ReplayBuffer
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.max_counter = []
        self.episode_rewards = []
        self.win_rates = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        # print('Run {} start'.format(num))
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                # win_rate, episode_reward = self.evaluate(all_gain,all_loss)
                self.evaluate(all_gain,all_loss)
                # print('win_rate is ', win_rate)
                # self.win_rates.append(win_rate)
                # self.episode_rewards.append(episode_reward)
            #     self.plt(num)
            episodes = []

            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                print("Generate episode {}".format(episode_idx))
                episode, episode_reward, info, win_number, all_gain, all_loss = self.rolloutWorker.generate_episode(episode_idx,0)
                episodes.append(episode)
                print('win_number:',win_number)
  
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0) #数组拼接

            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
            
    def evaluate(self,all_gain,all_loss):
        episode_rewards = 0
        win_number = 0
        x=[]
        for i in range(len(all_gain)):
            x.append(i+1)
        for epoch in range(self.args.evaluate_epoch):
            print('evaluate_epoch{}'.format(epoch))
            _, episode_reward, info, win_number,all_gain,all_loss = self.rolloutWorker.generate_episode(epoch, win_number, evaluate=True)
            print('win_number',win_number)
            episode_rewards += episode_reward
        
        plt.figure()
        plt.plot(all_gain, lw = 1.5,label = 'reward')
        plt.plot(all_loss, lw = 1.5,label = 'loss')
        plt.grid(True)
        plt.legend(loc = 0) #图例位置自动
        plt.axis('tight')
        plt.xlabel('index')
        plt.ylabel('packets')
        plt.title('reawrd and loss')
        plt.show()

        # return win_number / (self.args.evaluate_epoch*self.args.episode_limit), episode_rewards / 2