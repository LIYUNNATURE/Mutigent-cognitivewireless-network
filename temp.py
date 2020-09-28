from runner import Runner
from env import Environment
from arguments import get_common_args, get_mixer_args
from rollout import RolloutWorker
from agent import Agents

if __name__ == '__main__':

    args = get_common_args()    
    args = get_mixer_args(args)
    agents = Agents(args)

    env = Environment(args)
    runner = Runner(env,args)

    runner.run(0)
    