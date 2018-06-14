import numpy as np
import ene
from utils.functions import noop


def train(lake, episode_size=2000, action_callback=noop, ene_mode='e-greedy'):
    Q_sizes = [np.prod(lake.state_shape), lake.action_size]
    Q = np.zeros(np.prod(Q_sizes), dtype=np.float).reshape(Q_sizes)
    ene_method = ene.modes[ene_mode]
    history = {
        'rewards': []
    }

    action_spec = {
        'count': lake.action_size,
        'generator': lambda s: Q[state],
    }

    for episode in range(episode_size):
        state = lake.reset()
        action = -1
        actual_action = -1
        done = False
        total_reward = 0
        learning_rate = .85

        while not done:
            action_callback(lake, Q, episode, state, action, actual_action)

            action = ene_method(episode, state, action_spec, history=history)

            actual_action, new_state, reward, done = lake.step(action)
            # TODO: See why this is not working
            # Q[state][actual_action] = (1 - learning_rate) * Q[state][actual_action] + learning_rate * (reward + 0.99 * np.max(Q[new_state]))
            Q[state][action] = (1 - learning_rate) * Q[state][action] + learning_rate * (reward + 0.99 * np.max(Q[new_state]))
            state = new_state
            total_reward += reward

            action_callback(lake, Q, episode, state, action, actual_action)

        history['rewards'].append(total_reward)

    history['Q'] = Q

    return history
