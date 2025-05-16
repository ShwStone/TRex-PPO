import torch
import utils
from agent import DinoPPO
from trex import TRexRunner
import pathlib
import matplotlib.pyplot as plt
import sys

actor_lr = 1e-4
critic_lr = 1e-4
num_episodes = 200
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 4
eps = 0.2
id = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    recover = eval(sys.argv[1])

    env = TRexRunner()
    input_shape = env.shape()
    agent = DinoPPO(input_shape, hidden_dim, 3, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    pathlib.Path('./record').mkdir(exist_ok=True)
    pathlib.Path('./models').mkdir(exist_ok=True)
    return_list = utils.train_ppo(env, agent, num_episodes, 'record', 'models', recover, id)
    rewards, ratio = zip(*return_list)

    plt.plot(rewards)
    plt.savefig(f'record/{id}-reward.png')
    plt.close()

    plt.plot(ratio)
    plt.savefig(f'record/{id}-ratio.png')
    plt.close()

    env.close()