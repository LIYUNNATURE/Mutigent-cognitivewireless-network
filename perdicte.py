def get_common_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args
def get_env(args):
    args.lifetime = 5
    args.user_amount = 1
    args.n_agents =4
    
    args.episode = 100
    args.lambda_possion = 13
    args.reward = 0
    args.loss_packet_total = 0
    args.loss_packet = 0
    args.n_actions = 2
    args.state_shape = []
    
    return args

class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        # if args.alg == 'qmix':
        #     self.policy = QMIX(args)
        # else:
        #     raise Exception("No such algorithm")
        # self.args = args
        print('Init Agents')

class RolloutWorker:
    def __init__(self, agents, args):
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

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # self.agents.policy.init_hidden(1)

agent = Agents(args)
