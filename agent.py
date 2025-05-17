import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DinoActor, DinoCritic
import utils
import os
import numpy as np

class DinoPPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(
            self, 
            input_shape: tuple, 
            hidden_dim: int, 
            action_dim: int, 
            actor_lr: float, 
            critic_lr: float, 
            mini_batch: int, 
            lmbda: float, 
            epochs: int, 
            eps: float, 
            gamma: float, 
            device: torch.device
            ):
        self.actor = DinoActor(input_shape, hidden_dim, action_dim).to(device)
        self.critic = DinoCritic(input_shape, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.mini_batch = mini_batch
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

        deltas, targets, old_log_probs = [], [], []
        with torch.no_grad():
            for i in range(0, batch, self.mini_batch): 
                reward = rewards[i:i+self.mini_batch]
                next_state = next_states[i:i+self.mini_batch]
                done = dones[i:i+self.mini_batch]
                state = states[i:i+self.mini_batch]
                action = actions[i:i+self.mini_batch]

                target = reward + self.gamma * self.critic(next_state) * (1 - done)
                delta = target - self.critic(state)
                old_log_prob = F.log_softmax(self.actor(state), dim=-1).gather(1, action)

                deltas.append(delta)
                targets.append(target)
                old_log_probs.append(old_log_prob)

        targets = torch.cat(targets).detach()
        deltas = torch.cat(deltas).cpu()
        old_log_probs = torch.cat(old_log_probs).detach()

        advantages = utils.compute_advantage(self.gamma, self.lmbda, deltas).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indice = torch.randperm(batch)
        actions = actions[indice]
        states = states[indice]
        old_log_probs = old_log_probs[indice]
        advantages = advantages[indice]
        targets = targets[indice]

        ratios = 0
        clip_range = self.eps * ((256 / batch) ** .5)

        for _ in range(self.epochs):
            for i in range(0, batch, self.mini_batch):
                action = actions[i:i+self.mini_batch]
                state = states[i:i+self.mini_batch]
                old_log_prob = old_log_probs[i:i+self.mini_batch]
                advantage = advantages[i:i+self.mini_batch]
                target = targets[i:i+self.mini_batch]

                log_prob = F.log_softmax(self.actor(state), dim=-1).gather(1, action)
                ratio = torch.exp(log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage  # 截断
                
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean(F.mse_loss(self.critic(state), target))

                ratios += torch.mean(ratio).item()
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                actor_loss.backward()
                critic_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        iters = ((batch - 1) // self.mini_batch + 1) * self.epochs
        return ratios / iters
    
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