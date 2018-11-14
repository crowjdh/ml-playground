from math import sqrt

import pygame
from pygame import surfarray
# noinspection PyUnresolvedReferences
from pygame.locals import *
import numpy as np

# noinspection PyUnresolvedReferences
SCREEN_RECT = Rect(0, 0, 80, 80)


# noinspection PyUnresolvedReferences
def should_quit(event=None):
    if event:
        return event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE

    for event in pygame.event.get():
        if event.type == QUIT or \
                (event.type == KEYDOWN and event.key == K_ESCAPE):
                    return True


# noinspection PyUnresolvedReferences
class RMO(pygame.sprite.Sprite):
    speed_limit = 5
    normal_speed = [3, 2]
    rand_speed_candidates = list(range(-4, -1)) + list(range(1, 4))
    size = (10, 10)
    update_freq = 12

    def __init__(self, color, initial_position=None, randomly_change_direction=False):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.Surface(RMO.size)
        self.image.fill(color)
        third = int(SCREEN_RECT.width / 3)
        if initial_position:
            self.initial_position = initial_position
        else:
            self.initial_position = (np.random.randint(third, 2 * third), np.random.randint(third, 2 * third))
        self.randomly_change_direction = randomly_change_direction
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.rect = RMO.make_rect(self.initial_position, RMO.size)
        self.speed = self.make_random_speed() if self.randomly_change_direction else list(RMO.normal_speed)
        self.frame_since_last_speed_update = 0

    @property
    def should_update_speed(self):
        return self.frame_since_last_speed_update % RMO.update_freq == 0

    @property
    def is_inside_along_x(self):
        return 0 < self.rect.x < SCREEN_RECT.right

    @property
    def is_inside_along_y(self):
        return 0 < self.rect.y < SCREEN_RECT.bottom

    def update(self):
        target_rect = self.rect.move(*self.speed)
        if not SCREEN_RECT.contains(target_rect):
            if not RMO.contains_rect_horizontally(SCREEN_RECT, target_rect):
                self.speed[0] *= -1
            if not RMO.contains_rect_vertically(SCREEN_RECT, target_rect):
                self.speed[1] *= -1
            target_rect = self.rect.move(*self.speed)
            self.frame_since_last_speed_update = 0
        elif self.randomly_change_direction and self.should_update_speed:
            self.speed = RMO.make_random_speed()
            self.frame_since_last_speed_update = 0
        self.rect = target_rect

        self.frame_since_last_speed_update += 1

    @staticmethod
    def make_random_speed():
        speed_x = np.random.choice(RMO.rand_speed_candidates)
        speed_y = np.random.choice(RMO.rand_speed_candidates)
        speed = sqrt(pow(speed_x, 2) + pow(speed_y, 2))
        if speed > RMO.speed_limit:
            ratio = speed / RMO.speed_limit
            speed_x /= ratio
            speed_y /= ratio

        return [speed_x, speed_y]

    @staticmethod
    def make_rect(center, size):
        width, height = size
        center_x, center_y = center

        left_offset = width / 2
        top_offset = height / 2

        left = center_x - left_offset
        top = center_y - top_offset

        return pygame.Rect(left, top, width, height)

    @staticmethod
    def contains_rect_horizontally(lhs, rhs):
        return lhs.left <= rhs.left and rhs.right <= lhs.right

    @staticmethod
    def contains_rect_vertically(lhs, rhs):
        return lhs.top <= rhs.top and rhs.bottom <= lhs.bottom


class Zombie(RMO):
    def __init__(self, **kwargs):
        RMO.__init__(self, (255, 127, 127), **kwargs)


class Player(RMO):
    def __init__(self, **kwargs):
        RMO.__init__(self, (0, 0, 0), **kwargs)

    def update(self):
        pass

    def move(self, dx, dy):
        self.rect.move_ip(dx * RMO.speed_limit, dy * RMO.speed_limit)
        self.rect.clamp_ip(SCREEN_RECT)


# noinspection PyUnresolvedReferences
class Dodge:
    bg_color = (255, 255, 255)
    font_color = (255, 0, 0)

    def __init__(self, queue=None, move_zombies_randomly=False, show_additional_information=False):
        self.queue = queue
        self.show_additional_information = show_additional_information

        pygame.init()

        win_style = 0
        best_depth = pygame.display.mode_ok(SCREEN_RECT.size, win_style, 32)
        self.screen = pygame.display.set_mode(SCREEN_RECT.size, win_style, best_depth)

        self.zombies = pygame.sprite.Group()
        self.render_updates = pygame.sprite.RenderUpdates()

        self.font = pygame.font.SysFont(None, 20)
        self.ticks_area = self.font.render("9999", True, Dodge.font_color, Dodge.bg_color).get_rect()
        self.tries_area = self.font.render("999", True, Dodge.font_color, Dodge.bg_color).get_rect()
        self.tries_area.top = self.ticks_area.bottom

        self.ticks = 0
        self.tries = 0

        Player.containers = self.render_updates
        Zombie.containers = self.zombies, self.render_updates

        self.background = pygame.Surface(SCREEN_RECT.size)
        self.background.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

        self.player = Player(initial_position=(int(SCREEN_RECT.width / 4) * 2, SCREEN_RECT.center[1]))
        Zombie(initial_position=(int(SCREEN_RECT.width / 4), int(SCREEN_RECT.height / 4)),
               randomly_change_direction=move_zombies_randomly)
        Zombie(initial_position=(int(SCREEN_RECT.width / 4) * 3, int(SCREEN_RECT.height / 4)),
               randomly_change_direction=move_zombies_randomly)
        Zombie(initial_position=(int(SCREEN_RECT.width / 4), int(SCREEN_RECT.height / 4) * 3),
               randomly_change_direction=move_zombies_randomly)
        Zombie(initial_position=(int(SCREEN_RECT.width / 4) * 3, int(SCREEN_RECT.height / 4) * 3),
               randomly_change_direction=move_zombies_randomly)

        self.clock = pygame.time.Clock()

    def start(self):
        while not should_quit():
            dx, dy = None, None
            if self.queue:
                # if self.queue.empty():
                #     continue
                # dx, dy = self.queue.get(block=False)

                # if not self.queue.empty():
                #     dx, dy = self.queue.get(block=False)

                dx, dy = self.queue.get(block=True)
            else:
                keystate = pygame.key.get_pressed()
                dx = keystate[K_RIGHT] - keystate[K_LEFT]
                dy = keystate[K_DOWN] - keystate[K_UP]
                # dx = 1
                # dy = 1
            collided_zombies = self.tick(dx, dy)
            if len(collided_zombies) > 0:
                self.tries += 1
                self.reset_game()

        pygame.quit()

    def tick(self, dx, dy):
        self.ticks += 1
        self.update_game_objects(dx, dy)
        self.update_screen()

        self.clock.tick()

        return pygame.sprite.spritecollide(self.player, self.zombies, dokill=False)

    def reset_game(self):
        self.ticks = 0

        self.reset_game_objects()
        self.update_screen()

    def reset_game_objects(self):
        self.player.reset()
        for zombie in self.zombies.sprites():
            zombie.reset()

    def update_game_objects(self, dx, dy):
        if dx or dy:
            self.player.move(dx, dy)

        self.render_updates.update()

    def update_screen(self):
        self.render_updates.clear(self.screen, self.background)
        dirty = self.render_updates.draw(self.screen)

        # TODO: Separate score area from game area
        if self.show_additional_information:
            ticks = self.font.render(str(self.ticks), True, Dodge.font_color, Dodge.bg_color)
            self.screen.fill(Dodge.bg_color, rect=self.ticks_area)
            self.screen.blit(ticks, self.ticks_area)

            tries = self.font.render(str(self.tries), True, Dodge.font_color, Dodge.bg_color)
            self.screen.fill(Dodge.bg_color, rect=self.tries_area)
            self.screen.blit(tries, self.tries_area)

            dirty.append(self.ticks_area)
            dirty.append(self.tries_area)

        pygame.display.update(dirty)


def random_dx_dy():
    return np.random.randint(-1, 1), np.random.randint(-1, 1)


def get_grayscale_frame():
    frame = surfarray.array3d(
        pygame.display.get_surface()).astype(np.uint8)
    frame = frame / 255
    frame = 0.21 * frame[:, :, 0] + 0.72 * frame[:, :, 1] + 0.07 * frame[:, :, 2]

    return frame


def ensure_draw():
    pygame.event.get()


def dummy_user(queue, dodge):
    while True:
        queue.put(random_dx_dy())


def run():
    dodge = Dodge()

    while not should_quit():
        dodge.tick(*random_dx_dy())
        _ = get_grayscale_frame()


def run_on_thread():
    from queue import Queue
    from threading import Thread

    queue = Queue(maxsize=1)
    dodge = Dodge(queue=queue)

    worker = Thread(target=dummy_user, args=(queue, dodge,))
    worker.setDaemon(True)
    worker.start()

    dodge.start()


def play():
    Dodge().start()


if __name__ == '__main__':
    # run()
    # run_on_thread()
    play()
