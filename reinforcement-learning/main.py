import os
import curses
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import keyboard

is_paused = False
step_once = False
delays = [0.5, 0.2, 0.1, 0.05, 0.01]
delay_idx = 2
train_options = {}
callback_cursor = None


def main():
    args = parse_arguments()

    if args.run_mode == 'train':
        run_train(args)
    elif args.run_mode == 'replay':
        run_replay(args)


def run_train(args):
    train_options['train'] = train = import_train_method(args)
    train_options['env'] = env = create_environment(args)

    preprocess_env(env, args)
    _train(train, env, args)

    plot_history()


def run_replay(args):
    from utils.replay_manager import ReplayManager

    env = create_environment(args)
    ReplayManager(args.name).replay(env, start_idx=args.start_episode, end_idx=args.end_episode)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', action='store',
                        choices=['frozen_lake', 'dodge'], default='frozen_lake',
                        help='Environment to train')

    sub_parsers = parser.add_subparsers(help='Mode')
    train_parser = sub_parsers.add_parser('t', help='Train mode')
    replay_parser = sub_parsers.add_parser('r', help='Replay mode')

    train_parser.set_defaults(run_mode='train')
    replay_parser.set_defaults(run_mode='replay')

    train_parser.add_argument('-i', dest='interactive', action='store_true',
                              help='Enable interactive mode')
    train_parser.add_argument('--env_mode', action='store',
                              choices=['d', 's'], default='d',
                              help='Whether the environment is stochastic or deterministic')
    train_parser.add_argument('--train', dest='train_method', action='store',
                              choices=['q', 'dqn', 'pg'], default='dqn',
                              help='Training method')

    replay_parser.add_argument('-n', '--name', dest='name', action='store', required=True,
                               help='Name of saved replay')
    replay_parser.add_argument('--start', dest='start_episode', action='store', default=None,
                               type=int, help='Episode to replay from')
    replay_parser.add_argument('--end', dest='end_episode', action='store', default=None,
                               type=int, help='Episode to replay until')

    return parser.parse_args()


def import_train_method(args):
    if args.train_method == 'q':
        from learn.q_learning import train
    elif args.train_method == 'dqn':
        from learn.dqn import train
    elif args.train_method == 'pg':
        from learn.policy_gradient import train
    else:
        raise NotImplementedError("Train method {} not implemented".format(args.train_method))
    return train


def create_environment(args):
    is_stochastic = args.run_mode == 'train' and args.env_mode is 's'

    if args.env == 'frozen_lake':
        from environments.frozen_lake import FrozenLake
        return FrozenLake(is_stochastic=is_stochastic)
    elif args.env == 'dodge':
        from environments.dodge import DodgeEnv
        # TODO: Haven't seen stochastic dodge environment converging
        return DodgeEnv(headless_mode=False, is_stochastic=is_stochastic)


def preprocess_env(env, args):
    if args.env != 'frozen_lake':
        return
    if args.train_method in ['dqn', 'pg']:
        env.reward_processor = lambda a, s, r, d: -1 if d and r != 1 else r
        # 1 + 1 + 1 + 1 + 1  +  1 + 1 + 1 - 1 - 1 = 6
        env.threshold = 0.6
    if args.train_method == 'pg':
        env.penalty_on_going_out = True


def _train(train, env, args):
    if args.interactive:
        curses.wrapper(train_and_draw)
    else:
        train_options['history'] = train(env)


def plot_history():
    history = train_options['history']
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
    std_scr = callback_cursor

    std_scr.clear()

    episode_str = 'episode: ' + str(episode)
    possible_actions = ['up', 'right', 'down', 'left']
    describe_action = lambda act: possible_actions[act] if act >= 0 else ''
    message = episode_str + ', state: ' + str(state)
    if action >= 0 and actual_action >= 0:
        action_str = describe_action(action)
        actual_action_str = describe_action(actual_action)
        message += ', action: ' + action_str + ', actual_action: ' + actual_action_str
    std_scr.addstr(0, 0, message)

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

            std_scr.addstr(box_start[0] + 1, box_start[1] + 1, first_line, curses.color_pair(1))

            std_scr.addstr(box_start[0] + 2, box_start[1] + 1, second_line_1, curses.color_pair(1))
            std_scr.addstr(box_start[0] + 2, box_start[1] + 7, second_line_2, curses.color_pair(2))
            std_scr.addstr(box_start[0] + 2, box_start[1] + 11, second_line_3, curses.color_pair(1))

            std_scr.addstr(box_start[0] + 3, box_start[1] + 1, third_line, curses.color_pair(1))

    height, width = std_scr.getmaxyx()
    if is_privileged_mode():
        std_scr.addstr(height - 2, 0, "<Up>: Faster, <Down>: Slower")
        std_scr.addstr(height - 1, 0, "<Space>: Pause, <Right or n>: Step, <Ctrl + c>: Exit")
    else:
        std_scr.attron(curses.color_pair(2))
        std_scr.addstr(height - 2, 0, "Run as privileged mode for more controls.")
        std_scr.attroff(curses.color_pair(2))
        std_scr.addstr(height - 1, 0, "<Ctrl + c>: Exit")

    std_scr.refresh()

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


def train_and_draw(std_scr):
    std_scr.clear()
    std_scr.refresh()

    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    std_scr.clear()

    hook_keyboard_event()

    global callback_cursor
    callback_cursor = std_scr
    train = train_options['train']
    env = train_options['env']

    train_options['history'] = train(env, action_callback=action_callback)

    height, width = std_scr.getmaxyx()
    std_scr.addstr(height - 2, 0, ' ' * (width - 1))
    std_scr.addstr(height - 1, 0, ' ' * (width - 1))
    std_scr.addstr(height - 1, 0, "Done all episodes. Press any key to exit")
    _ = std_scr.getch()


if __name__ == "__main__":
    main()
