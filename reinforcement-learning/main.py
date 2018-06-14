import os
import curses
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import keyboard

from environments.frozen_lake import FrozenLake
# noinspection PyUnresolvedReferences
from learn.q_learning import train as q_learning_train
from learn.dqn import train as dqn_train

is_paused = False
step_once = False
delays = [0.5, 0.2, 0.1, 0.05, 0.01]
delay_idx = 2
training_methods = {
    'q': q_learning_train,
    'dqn': dqn_train
}


def main():
    args = parse_arguments()

    global train, lake
    train = training_methods[args.method]
    lake = FrozenLake(is_slippery=False)
    # lake = FrozenLake(is_slippery=True)

    if args.method == 'dqn':
        lake.reward_processor = lambda a, s, r, d: -1 if d and r != 1 else r

    if args.interactive:
        curses.wrapper(train_and_draw)
    else:
        global history
        history = train(lake)

    plot_history()


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='interactive', action='store_true',
                        help='Enable interactive mode')
    parser.add_argument('--method', action='store',
                        choices=['q', 'dqn'], default='dqn',
                        help='Mode')
    return parser.parse_args()


def plot_history():
    global history
    if not history:
        return

    print(history['Q'])

    rewards = np.array(history['rewards'])
    success_rate = rewards.sum() / len(rewards)
    print("success_rate: {}\nSee plot for more detail.".format(success_rate))

    for i, key in enumerate(history):
        if key == 'Q':
            continue
        plt.subplot(len(history), 1, i + 1)
        plt.ylabel(key)
        plt.plot(history[key])
    plt.show()


def action_callback(lake, Q, episode, state, action, actual_action):
    state = lake.unflatten_state(state)
    Q = Q.reshape(list(lake.state_shape) + [lake.action_size])

    global callback_cursor
    stdscr = callback_cursor

    stdscr.clear()

    episode_str = 'episode: ' + str(episode)
    possible_actions = ['up', 'right', 'down', 'left']
    describe_action = lambda act: possible_actions[act] if act >= 0 else ''
    message = episode_str + ', state: ' + str(state)
    if action >= 0 and actual_action >= 0:
        action_str = describe_action(action)
        actual_action_str = describe_action(actual_action)
        message += ', action: ' + action_str + ', actual_action: ' + actual_action_str
    stdscr.addstr(0, 0, message)

    for r in range(Q.shape[0]):
        for c in range(Q.shape[1]):
            if state == (r, c):
                val = 'AI'
            elif (r, c) in lake.pitfalls:
                val = 'XX'
            elif (r, c) == lake.goal:
                val = '!!'
            else:
                val = ''

            up, right, down, left = Q[r, c]
            first_line = '{:^16.1f}'.format(up)
            second_line_1 = '{:^6.1f}'.format(left)
            second_line_2 = '{:^4s}'.format(val)
            second_line_3 = '{:^6.1f}'.format(right)
            third_line = '{:^16.1f}'.format(down)

            box_start = (r * 4, c * 16)

            stdscr.addstr(box_start[0] + 1, box_start[1] + 1, first_line, curses.color_pair(1))

            stdscr.addstr(box_start[0] + 2, box_start[1] + 1, second_line_1, curses.color_pair(1))
            stdscr.addstr(box_start[0] + 2, box_start[1] + 7, second_line_2, curses.color_pair(2))
            stdscr.addstr(box_start[0] + 2, box_start[1] + 11, second_line_3, curses.color_pair(1))

            stdscr.addstr(box_start[0] + 3, box_start[1] + 1, third_line, curses.color_pair(1))

    height, width = stdscr.getmaxyx()
    if is_privileged_mode():
        stdscr.addstr(height - 2, 0, "<Up>: Faster, <Down>: Slower")
        stdscr.addstr(height - 1, 0, "<Space>: Pause, <Right or n>: Step, <Ctrl + c>: Exit")
    else:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(height - 2, 0, "Run as privileged mode for more controls.")
        stdscr.attroff(curses.color_pair(2))
        stdscr.addstr(height - 1, 0, "<Ctrl + c>: Exit")

    stdscr.refresh()

    sleep(delays[delay_idx])
    while is_paused:
        global step_once
        if step_once:
            step_once = False
            break
        continue


def is_privileged_mode():
    try:
        return os.getuid() == 0
    except AttributeError:
        return False


def hook_keyboard_event():
    if is_privileged_mode():
        keyboard.on_release(handle_command)


def handle_command(event):
    global delay_idx, is_paused, step_once
    key_name = event.name
    if key_name == 'space':
        is_paused = not is_paused
    elif is_paused and key_name in ['right', 'n']:
        step_once = True
    elif key_name in ['down', 'up']:
        if key_name == 'down':
            delay_idx -= 1
        elif key_name == 'up':
            delay_idx += 1

        delay_idx = max(min(delay_idx, len(delays) - 1), 0)


def train_and_draw(stdscr):
    stdscr.clear()
    stdscr.refresh()

    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    stdscr.clear()

    hook_keyboard_event()

    global history, callback_cursor, lake, train
    callback_cursor = stdscr
    history = train(lake, action_callback=action_callback)

    height, width = stdscr.getmaxyx()
    stdscr.addstr(height - 2, 0, ' ' * (width - 1))
    stdscr.addstr(height - 1, 0, ' ' * (width - 1))
    stdscr.addstr(height - 1, 0, "Done all episodes. Press any key to exit")
    k = stdscr.getch()


if __name__ == "__main__":
    main()
