import sys, os
import curses
from time import sleep
import matplotlib.pyplot as plt
import keyboard

import model

is_paused = False
step_once = False
delays = [0.5, 0.2, 0.1, 0.05, 0.01]
delay_idx = 2


def main():
    curses.wrapper(train_and_draw)


def pre_action_callback(stdscr, board, Q, episode, state, action):
    stdscr.clear()

    episode_str = 'episode: ' + str(episode)
    action_str = ['up', 'right', 'down', 'left'][action] if 'action' in locals() and action else ''
    stdscr.addstr(0, 0, episode_str + ', action: ' + str(action) + ', state: ' + str(state) + ', direction: ' + action_str)

    for r in range(Q.shape[0]):
        for c in range(Q.shape[1]):
            if state == (r, c):
                val = 'AI'
            elif board[r, c] == -1:
                val = 'XX'
            elif board[r, c] == 1:
                val = '!!'
            else:
                val = ''
            up, right, down, left = Q[r, c]
            first_line = '  {:2d}  '.format(up)
            second_line_1 = '{:2d}'.format(left)
            second_line_2 = '{:2s}'.format(val)
            second_line_3 = '{:2d}'.format(right)
            third_line = '  {:2d}  '.format(down)

            box_start = (r * 4, c * 8)
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(box_start[0] + 1, box_start[1] + 1, first_line)
            stdscr.addstr(box_start[0] + 2, box_start[1] + 1, second_line_1)
            stdscr.attroff(curses.color_pair(1))

            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(box_start[0] + 2, box_start[1] + 3, second_line_2)
            stdscr.attroff(curses.color_pair(2))

            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(box_start[0] + 2, box_start[1] + 5, second_line_3)
            stdscr.addstr(box_start[0] + 3, box_start[1] + 1, third_line)
            stdscr.attroff(curses.color_pair(1))

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

    stdscr.clear()
    height, width = stdscr.getmaxyx()

    hook_keyboard_event()

    # Train
    rewards = model.train(stdscr, pre_action_callback)

    height, width = stdscr.getmaxyx()
    stdscr.addstr(height - 2, 0, ' ' * (width - 1))
    stdscr.addstr(height - 1, 0, ' ' * (width - 1))
    stdscr.addstr(height - 1, 0, "Done all episodes. Press any key to exit")
    k = stdscr.getch()


if __name__ == "__main__":
    main()
