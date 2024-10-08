import gymnasium as gym
import tensorflow as tf 
import numpy as np
import random

from tqdm import tqdm
from Config import deep_q_network_props
from collections import deque
from typing import Dict



env = gym.make('Taxi-v3')
state_size = env.observation_space.n    # 500 states
action_size = env.action_space.n        # 6 actions

#Store transitions in a replay buffer
memory = deque(maxlen=deep_q_network_props.memory_size)

#Model definition
def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=state_size, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=deep_q_network_props.learning_rate), loss='mse')
    return model

#Action choice
def select_action(state, model: tf.keras.Sequential, info: Dict, exploration_rate: float):
    action_mask = info.get('action_mask')        
    valid_actions = np.where(action_mask == 1)[0]

    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(valid_actions)      #Exploration: pick a random action
    
    q_values = model.predict(np.identity(state_size)[state:state+1])
    return np.argmax(q_values[0])                   #Exploitation: choose the action with the highest Q-Value


def train_network(model):
    if len(memory) < deep_q_network_props.train_start:
        return
    
    batch = random.sample(memory, deep_q_network_props.batch_size)    
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.identity(state_size)[np.array(states)]        
    next_states = np.identity(state_size)[np.array(next_states)] 

    actions = np.array(actions)
    rewards = np.array(rewards)    
    dones = np.array(dones)

    current_q_values = model.predict(states)
    next_q_values = model.predict(next_states)

    targets = np.copy(current_q_values)

    for i in range(deep_q_network_props.batch_size):
        if dones[i]:
            targets[i][actions[i]] = rewards[i]
        else:
            targets[i][actions[i]] = rewards[i] + deep_q_network_props.discount_factor * np.max(next_q_values[i])

    model.fit(states, targets, batch_size=deep_q_network_props.batch_size, epochs=1, verbose=0)    



model = create_model()
epsilon = deep_q_network_props.exploration_rate
success_table = {}


for episode in tqdm(range(deep_q_network_props.num_episodes), desc="Training "):
    state, info = env.reset()
    completed = False
    current_step = 0

    while not completed and current_step < deep_q_network_props.max_steps_per_episode:
        action = select_action(state, model, info, epsilon)

        next_state, reward, terminated, truncated, new_info = env.step(action)

        #Store the experience
        memory.append((state, action, reward, next_state, terminated))

        state = next_state
        info = new_info
        current_step += 1

        if terminated or truncated:
            if not truncated:
                success_table[episode] = True
            completed = True
        
    
    #Decay the exploration rate
    epsilon = max(deep_q_network_props.minimum_exploration_rate, epsilon * deep_q_network_props.exploration_rate_decay)    

    if (episode + 1) % deep_q_network_props.train_frequency == 0 and len(memory) >= deep_q_network_props.batch_size:
        train_network(model)



print(f"Total number of successfull episodes: {len(success_table)}")
print(success_table)
env.close()


model.save_weights('./CheckPoints/cp.ckpt')













