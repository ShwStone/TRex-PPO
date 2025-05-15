import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DinoActor, DinoCritic
import utils
import os
import numpy as np

class DinoPPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = DinoActor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = DinoCritic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(
            np.expand_dims(state, 0), 
            dtype=torch.float
        ).to(self.device)
        logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(
            np.array(transition_dict['states']), 
            dtype=torch.float
        ).to(self.device)
        actions = torch.tensor(
            np.array(transition_dict['actions'])
        ).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            np.array(transition_dict['rewards']), 
            dtype=torch.float
        ).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            np.array(transition_dict['next_states']), 
            dtype=torch.float
        ).to(self.device)
        dones = torch.tensor(
            np.array(transition_dict['dones']), 
            dtype=torch.float
        ).view(-1, 1).to(self.device)

        batch = states.size()[0]

        deltas, targets = [], []
        for i in range(0, batch, 32): 
            reward = rewards[i:i+32]
            next_state = next_states[i:i+32]
            done = dones[i:i+32]
            state = states[i:i+32]

            target = reward + self.gamma * self.critic(next_state) * (1 - done)
            delta = target - self.critic(state)
            deltas.append(delta)
            targets.append(target)

        targets = torch.cat(targets).detach()
        deltas = torch.cat(deltas).cpu()

        advantages = utils.compute_advantage(self.gamma, self.lmbda, deltas).to(self.device)
        old_log_probs = F.log_softmax(self.actor(states), dim=-1).gather(1, actions).detach()

        indice = torch.randperm(batch)
        actions = actions[indice]
        states = states[indice]
        old_log_probs = old_log_probs[indice]
        advantages = advantages[indice]
        targets = targets[indice]

        for _ in range(self.epochs):
            for i in range(0, batch, 32):
                action = actions[i:i+32]
                state = states[i:i+32]
                old_log_prob = old_log_probs[i:i+32]
                advantage = advantages[i:i+32]
                target = targets[i:i+32]

                log_prob = F.log_softmax(self.actor(state), dim=-1).gather(1, action)
                ratio = torch.exp(log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean(F.mse_loss(self.critic(state), target))
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                actor_loss.backward()
                critic_loss.backward()
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
    
    def save(self, path) :
        actor_path = os.path.join(path, 'actor.pth')
        critic_path = os.path.join(path, 'critic.pth')
        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)

    def load(self, path) :
        actor_path = os.path.join(path, 'actor.pth')
        critic_path = os.path.join(path, 'critic.pth')
        self.actor = torch.load(actor_path, weights_only=False)
        self.critic = torch.load(critic_path, weights_only=False)