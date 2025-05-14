import torch
import utils
from agent import DinoPPO
from trex import TRexRunner
import pathlib

actor_lr = 1e-4
critic_lr = 1e-4
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 4
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    env = TRexRunner()
    input_shape = env.shape()
    agent = DinoPPO(input_shape, hidden_dim, 3, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    pathlib.Path('./record').mkdir(exist_ok=True)
    pathlib.Path('./models').mkdir(exist_ok=True)
    return_list = utils.train_ppo(env, agent, num_episodes, 'record', 'models', False)
    env.close()