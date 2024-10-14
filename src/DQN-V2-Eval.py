import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from tqdm import trange


class RLModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(RLModel, self).__init__()

        self.emb = nn.Embedding(input_size, 4)
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.emb(x)
        x = nn.functional.relu(self.l1(x))    
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x
    

def choose_action(model, state, info):
    action_mask = info.get('action_mask')   

    with torch.no_grad():                               
        predicted = model(torch.tensor([state], device=torch.device('cpu')))        
        q_values = predicted.cpu().numpy()[0]
        q_values[action_mask == 0] = -float('inf')
        action = q_values.argmax()        
        return action 



env = gym.make("Taxi-v3", render_mode='human')
state_size = env.observation_space.n        
action_size = env.action_space.n

model = RLModel(state_size, action_size).to(torch.device('cpu'))
model.load_state_dict(torch.load(Path('./src/DQN-V2.pth')))
model.eval()


for episode in range(10):
    print(f"Episode number: {episode + 1}")
    state, info = env.reset()
    total_reward = 0

    progress_bar = trange(0, 200, initial=0, total=200)

    for _ in progress_bar:
        action = choose_action(model, state, info)

        print(f"Selected action: {action}")

        next_state, reward, done, truncated, new_info = env.step(action)
        total_reward += reward

        state = next_state
        info = new_info

        if done or truncated:
            print(f"Total reward {total_reward}")
            if truncated:
                print(f"Insuccess for episode {episode + 1}")
            else:
                print(f"Success for episode {episode + 1}")

            break

env.close()