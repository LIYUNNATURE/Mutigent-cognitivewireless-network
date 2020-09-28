from runner import Runner
from env import Environment
from arguments import get_common_args, get_mixer_args

if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        args = get_mixer_args(args)

        env = Environment(args)
        runner = Runner(env,args)

        runner.run(i) #学习 
        
        win_rate, _ = runner.evaluate()#评估
        print('The win rate of {} is  {}'.format(args.alg, win_rate))
        break
        # env.close()
  

