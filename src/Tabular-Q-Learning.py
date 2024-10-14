import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from Config import tabular_q_learning_props
from tqdm import tqdm


env = gym.make('Taxi-v3')

#Initialize Q-Table. It will be a table with dimension 500 x 6
q_table = np.zeros((env.observation_space.n, env.action_space.n))

#Get configuration parameters
num_episodes = tabular_q_learning_props.num_episodes
max_steps = tabular_q_learning_props.max_steps_per_episode
alpha = tabular_q_learning_props.learning_rate
gamma = tabular_q_learning_props.discount_factor
epsilon = tabular_q_learning_props.exploration_rate
epsilon_min = tabular_q_learning_props.minimum_exploration_rate
epsilon_decay = tabular_q_learning_props.exploration_rate_decay

success_table = {}
total_rewards_list = []

for episode in tqdm(range(num_episodes), desc="Executing"):
    state, info = env.reset() #Start the environment and obtain the initial state    
    completed = False
    current_step = 0
    total_reward = 0

    while not completed and current_step < max_steps:
        action_mask = info.get('action_mask')        
        valid_actions = np.where(action_mask == 1)[0]

        if np.random.uniform(0, 1) < epsilon:    
            action = np.random.choice(valid_actions)  #Exploration: pick a random action 
        else:
            q_values_valid_actions = q_table[state, valid_actions]
            action = valid_actions[np.argmax(q_values_valid_actions)]  #Exploitation: choose the action with the highest Q-Value

        next_state, reward, done, truncated, new_info = env.step(action)     #Execute the action
        total_reward += reward

        #Update the Q-Value
        best_next_action = np.argmax(q_table[next_state, :])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

        state = next_state
        info = new_info
        current_step += 1

        if done or truncated:
            if not truncated:                
                success_table[episode] = True
            completed = True
       
    epsilon = epsilon_min + (tabular_q_learning_props.exploration_rate - epsilon_min) * np.exp(-episode / epsilon_decay)        
    total_rewards_list.append(total_reward)


print(f"Total number of successfull episodes: {len(success_table)}")
print(success_table)
        
env.close()

new_env = gym.make('Taxi-v3', render_mode="human")

for episode in range(10):
    print(f"Episode number: {episode + 1}")
    state, info = new_env.reset()

    for _ in tqdm(range(300)):
        action_mask = info.get('action_mask')
        valid_actions = np.where(action_mask == 1)[0]

        q_values_valid_actions = q_table[state, valid_actions]
        action = valid_actions[np.argmax(q_values_valid_actions)]
                    
        next_state, reward, terminated, truncated, new_info = new_env.step(action)   

        state = next_state
        info = new_info

        if terminated or truncated:            
            if truncated:
                print(f"Insuccess for episode {episode + 1}")
            else: 
                print(f"Success for episode {episode + 1}")

            break

new_env.close()
        

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))        
plt.title("Training result")

ax1.set_xlabel('Episode')
ax1.set_ylabel('Rewards')
ax1.set_ylim(-230, 250)
ax1.set_title("Rewards")
ax1.plot(total_rewards_list, color="red")

plt.show()














