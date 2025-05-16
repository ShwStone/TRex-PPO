from tqdm import tqdm
import os
import torch
import numpy as np

def train_ppo(env, agent, num_episodes, record_prefix, models_prefix, resume, id):
    return_list = []
    if resume:
        agent.load(models_prefix)
    with tqdm(total=num_episodes) as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            record = True
            record_path = os.path.join(record_prefix, f'{id}-{i_episode}.gif')
            state = env.begin()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action, record=record, record_path=record_path)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                state = next_state
                episode_return += reward
            ratio = agent.update(transition_dict)
            return_list.append((episode_return, ratio))
            pbar.set_postfix({'episode': f'{i_episode}', 'reward': f'{episode_return}', 'ratio': f'{ratio}'})
            pbar.update(1)
            if record:
                agent.save(models_prefix)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)