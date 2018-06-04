import numpy as np
from utils.math_utils import rand_argmax


def select_with_noise(episode, possible_actions, history):
    noise = np.random.randn(len(possible_actions)) / (episode + 1)

    action = np.argmax(possible_actions + noise)
    action_wo_noise = rand_argmax(possible_actions)

    if 'noised' not in history:
        history['noised'] = []
        history['noise'] = []
    history['noised'].append(action != action_wo_noise)
    history['noise'].append(noise)

    return action


def select_with_e_greedy(episode, possible_actions, history):
    e = 1. / ((episode / 100) + 1)

    choose_action_randomly = (np.random.rand(1) < e)[0]

    if 'randomly_selected' not in history:
        history['randomly_selected'] = []
        history['e'] = []
    history['randomly_selected'].append(choose_action_randomly)
    history['e'].append(e)

    if choose_action_randomly:
        return np.random.choice(len(possible_actions))
    else:
        return rand_argmax(possible_actions)


modes = {
    'noise': select_with_noise,
    'e-greedy': select_with_e_greedy,
}
