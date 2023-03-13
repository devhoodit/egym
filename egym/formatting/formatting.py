import numpy as np

def wrap_print_cycle(print_format: str, n_epi, reward_history, iter_history):
    return print_format.format(
        episode=n_epi,
        average_reward=np.average(reward_history),
        max_reward=max(reward_history),
        min_reward=min(reward_history),
        std_reword=np.std(reward_history),
        average_iter_length=np.average(iter_history),
        max_iter_length=max(iter_history),
        min_iter_length=min(iter_history),
        std_iter_length=np.std(iter_history)
    )
