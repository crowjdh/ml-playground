import numpy as np
from utils import rand_argmax


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


ene_modes = {
    'noise': select_with_noise,
    'e-greedy': select_with_e_greedy,
}


def train(lake, episode_size = 2000, action_callback=None, ene_mode='noise'):
    action_size = len(lake.action_position_map)
    Q_sizes = list(lake.lake_size) + [action_size]
    Q = np.zeros(np.prod(Q_sizes), dtype=np.float).reshape(Q_sizes)
    ene_method = ene_modes[ene_mode]
    history = {
        'rewards': []
    }

    for episode in range(episode_size):
        state = lake.reset()
        action = -1
        actual_action = -1
        done = False
        total_reward = 0
        learning_rate = .85

        while not done:
            if action_callback:
                action_callback(lake.lake, Q, episode, state, action, actual_action)

            possible_actions = Q[state]
            action = ene_method(episode, possible_actions, history)

            actual_action, new_state, reward, done = lake.step(action)
            # TODO: See why this is not working
            # Q[state][actual_action] = (1 - learning_rate) * Q[state][actual_action] + learning_rate * (reward + 0.99 * np.max(Q[new_state]))
            Q[state][action] = (1 - learning_rate) * Q[state][action] + learning_rate * (reward + 0.99 * np.max(Q[new_state]))
            state = new_state
            total_reward += reward

            if action_callback:
                action_callback(lake.lake, Q, episode, state, action, actual_action)

        history['rewards'].append(total_reward)

    history['Q'] = Q

    return history
