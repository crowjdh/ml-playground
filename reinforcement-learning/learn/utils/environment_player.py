import numpy as np


def simulate_play(env, network, count=10):
    for i in range(count):
        state = env.reset()
        done = False
        reward_sum = 0

        while not done:
            action = np.argmax(network.predict(state))
            actual_action, new_state, reward, done = env.step(action)
            reward_sum += reward
            state = new_state

        print("reward_sum: {}".format(reward_sum))
