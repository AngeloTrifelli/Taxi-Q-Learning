import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import gymnasium as gym 
import random

from collections import deque
from tqdm import trange
from Config import deep_q_network_props


# -------- MODEL DEFINITION ---------#
class RLModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(RLModel, self).__init__()

        self.embedding = nn.Embedding(input_size, 4)
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = nn.functional.relu(self.fc1(x))    
        x = nn.functional.relu(self.fc2(x))
        x = self.output(x)
        return x


# ------------ FUNCTIONS -------------- #
def choose_action(model, state: int, exploration_rate: float):    
    if np.random.uniform(0, 1) < exploration_rate:                       
        action = env.action_space.sample()        # Exploration: pick a random action
    else:
        with torch.no_grad():                                
            q_values = model(torch.tensor([state], device=torch.device('cpu')))     #Exploitation: pick the action with the highest q_value
            action = q_values.max(1)[1].item()    

    return action
    
def train_model(model, target_model, optimizer, loss_fn, memory):
    if len(memory) < deep_q_network_props.network.batch_size:
        return          

    batch = random.sample(memory, deep_q_network_props.network.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    state_batch = torch.tensor(states, device=torch.device('cpu'))
    action_batch = torch.tensor(actions, device=torch.device('cpu'), dtype=torch.long) 
    reward_batch = torch.tensor(rewards, device=torch.device('cpu'))
    next_state_batch = torch.tensor(next_states, device=torch.device('cpu'))
    done_batch = torch.tensor(dones, device=torch.device('cpu'), dtype=torch.bool)

     # Compute predicted Q values
    predicted_q_value = model(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # Use Bellman equation to compute new expected values
    expected_q_values = reward_batch + (torch.logical_not(done_batch) * target_model(next_state_batch).max(1)[0] * 0.99)

    # Compute loss
    loss = loss_fn(predicted_q_value, expected_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
   

def decay_learning_rate(optimizer, episode):
    minimum_lr = deep_q_network_props.network.minimum_lr
    new_lr = minimum_lr + (deep_q_network_props.network.initial_lr - minimum_lr) * np.exp(- episode / deep_q_network_props.network.lr_decay)
    for elem in optimizer.param_groups:
        elem['lr'] = new_lr


def plot_chart(total_reward_list, epsilon_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))        
    plt.subplots_adjust(wspace=0.5)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rewards')
    ax1.set_ylim(-1000, 250)
    ax1.set_title("Episode vs total reward")
    ax1.plot(total_reward_list, color="red")

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon decay')
    ax2.plot(epsilon_list, color="C2")

    plt.show()



# -------------- INITIALIZATION ------------#
env = gym.make('Taxi-v3')
state_size = env.observation_space.n        # 500 states
action_size = env.action_space.n            # 6 actions

model = RLModel(state_size, action_size).to(torch.device('cpu'))
optimizer = torch.optim.Adam(model.parameters(), lr=deep_q_network_props.network.initial_lr)
loss_fn = torch.nn.functional.smooth_l1_loss

target_model = RLModel(state_size, action_size).to(torch.device('cpu'))
target_model.load_state_dict(model.state_dict())
target_model.eval()


memory = deque(maxlen=deep_q_network_props.rl.memory_size)
epsilon = deep_q_network_props.rl.exploration_rate
minimum_epsilon = deep_q_network_props.rl.minimum_exploration_rate
epsilon_decay = deep_q_network_props.rl.exploration_rate_decay

total_reward_list = []
epsilon_list = []


# -------------- TRAINING  --------------#
progress_bar = trange(0, deep_q_network_props.env.num_episodes)
successful_episodes = []


for episode in progress_bar:
    state, _ = env.reset()
    total_reward = 0
    
    if len(memory) >= deep_q_network_props.rl.train_start:
        epsilon = minimum_epsilon + (deep_q_network_props.rl.exploration_rate - minimum_epsilon) * np.exp(-episode / epsilon_decay)        

    for current_step in range(0, deep_q_network_props.env.max_steps_per_episode):        
        action = choose_action(model, state, epsilon)    
    
        next_state, reward, done, truncated, _ = env.step(action)

        if len(memory) > deep_q_network_props.rl.memory_size:
            memory.popleft()

        memory.append([*[state, action, reward, next_state, done]])
        
        if len(memory) >= deep_q_network_props.rl.train_start:
            train_model(model, target_model, optimizer, loss_fn, memory)
            decay_learning_rate(optimizer, episode)
            
        state = next_state
        total_reward += reward

        if done and not truncated:
            successful_episodes.append(episode)

        done = done or (current_step == deep_q_network_props.env.max_steps_per_episode - 1)

        if done or truncated:        
            total_reward_list.append(total_reward)
            epsilon_list.append(epsilon)
            
            progress_bar.set_postfix({
                "reward": total_reward,
                "epsilon": epsilon
            })
            break

    if episode % deep_q_network_props.rl.target_network_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
        


plot_chart(total_reward_list, epsilon_list)
print(f"Total number of successful episodes: {len(successful_episodes)}")
torch.save(model.state_dict(), './models/DQN-V2.pth')