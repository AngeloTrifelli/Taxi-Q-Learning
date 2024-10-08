from types import SimpleNamespace

tabular_q_learning_props = SimpleNamespace(
    learning_rate = 0.1,
    discount_factor = 0.99,
    exploration_rate = 1.0,
    exploration_rate_decay = 0.995,
    minimum_exploration_rate = 0.01,
    num_episodes = 2000,
    max_steps_per_episode = 400
)

deep_q_network_props = SimpleNamespace(
    learning_rate = 0.001,              #
    discount_factor = 0.99,             
    exploration_rate = 1.0,             #Initial exploration rate
    exploration_rate_decay = 0.995,     #Decay after each selected action
    minimum_exploration_rate = 0.01,    
    num_episodes = 2000,                #Total number of episodes that will be executed
    max_steps_per_episode= 400,         #Maximum number of steps executed for each episode
    memory_size = 2000,                 #Maximum size of the replay buffer
    train_start = 1000,                 #Minimum number of elements required in the buffer memory in order to start the training 
    train_frequency = 10,       
    batch_size = 64                 
)