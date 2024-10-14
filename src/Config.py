from types import SimpleNamespace

tabular_q_learning_props = SimpleNamespace(
    learning_rate = 0.1,
    discount_factor = 0.99,
    exploration_rate = 1.0,
    exploration_rate_decay = 400,
    minimum_exploration_rate = 0.01,
    num_episodes = 3000,
    max_steps_per_episode = 400
)

deep_q_network_props = SimpleNamespace(
    network = SimpleNamespace(
        initial_lr = 0.001,
        minimum_lr = 0.0001,
        lr_decay = 5000,
        batch_size = 128                
    ),
    env = SimpleNamespace(
        num_episodes = 7000,
        max_steps_per_episode = 100        
    ),
    rl = SimpleNamespace(
        discount_factor = 0.99,
        exploration_rate = 1,
        minimum_exploration_rate = 0.1,
        exploration_rate_decay = 400,
        memory_size = 5000,
        train_start = 1000,
        target_network_update_freq = 20
    )                    
)