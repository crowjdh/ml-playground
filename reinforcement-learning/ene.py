import numpy as np
from utils.math_utils import rand_argmax


def select_with_noise(episode, state, action_spec, history=None):
    noise = np.random.randn(action_spec['count']) / (episode + 1)
    actions = action_spec['generator'](state)

    action = np.argmax(actions + noise)

    action_wo_noise = rand_argmax(actions)
    fill_history(history, ['noised', action != action_wo_noise], ['noise', noise])

    return action


def select_with_e_greedy(episode, state, action_spec, history=None):
    e = 1. / ((episode / 10) + 1)
    e = max(e, 0.03)

    choose_action_randomly = (np.random.rand(1) < e)[0]

    fill_history(history, ['randomly_selected', choose_action_randomly], ['e', e])

    if choose_action_randomly:
        return np.random.choice(action_spec['count'])
    else:
        return rand_argmax(action_spec['generator'](state))


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
