import numpy as np
from utils.math_utils import rand_argmax


def select_with_noise(episode, possible_actions, history=None):
    noise = np.random.randn(len(possible_actions)) / (episode + 1)

    action = np.argmax(possible_actions + noise)

    action_wo_noise = rand_argmax(possible_actions)
    fill_history(history, ['noised', action != action_wo_noise], ['noise', noise])

    return action


def select_with_e_greedy(episode, possible_actions, history=None):
    e = 1. / ((episode / 10) + 1)

    choose_action_randomly = (np.random.rand(1) < e)[0]

    fill_history(history, ['randomly_selected', choose_action_randomly], ['e', e])

    if choose_action_randomly:
        return np.random.choice(len(possible_actions))
    else:
        return rand_argmax(possible_actions)


def fill_history(history, *entries):
    if not history:
        return

    for key, value in entries:
        if key not in history:
            history[key] = []
        history[key].append(value)


modes = {
    'noise': select_with_noise,
    'e-greedy': select_with_e_greedy,
}
