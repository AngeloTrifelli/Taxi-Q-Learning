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