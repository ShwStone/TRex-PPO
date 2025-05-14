import random

import pygame

from events import ADD_ENEMY, KILL_ENEMY
from speed import Speed
from sprites.base import SpriteBase


class Cactus(SpriteBase):
    def __init__(self, image: pygame.Surface, speed: Speed) -> None:
        super().__init__()
        self.image = image
        self.pos_x = 0
        self.speed = speed
        self.spaned = False

    def update(self, screen: pygame.Surface):
        w_screen, h_screen = screen.get_size()
        w_image, h_image = self.image.get_size()
        self.pos_x += self.speed.value

        if self.pos_x >= w_screen / 2 + w_image:
            if not self.spaned:
                self.spaned = True
                pygame.event.post(pygame.event.Event(ADD_ENEMY))

        if self.pos_x >= w_screen + w_image:
            self.kill()
            pygame.event.post(pygame.event.Event(KILL_ENEMY))

        self.x = -self.pos_x + w_screen
        self.y = h_screen - h_image
        self.render(screen)


class Pterodactyl(SpriteBase):
    def __init__(self, images: list[pygame.Surface], speed: Speed) -> None:
        super().__init__()
        self.images = images
        self.pos_x = 0
        self.y = random.randint(100, 200)
        self.speed = speed
        self.index = 0
        self.num_flap = 10
        self.spaned = False

    @property
    def image(self) -> pygame.Surface:
        return self.images[self.index // self.num_flap]

    def update(self, screen: pygame.Surface):
        w_screen, h_screen = screen.get_size()
        w_image, h_image = self.image.get_size()
        self.pos_x += self.speed.value
        self.index += 1
        self.index %= self.num_flap * 2

        if self.pos_x >= w_screen / 2 + w_image:
            if not self.spaned:
                self.spaned = True
                pygame.event.post(pygame.event.Event(ADD_ENEMY))

        if self.pos_x >= w_screen + w_image:
            self.kill()
            pygame.event.post(pygame.event.Event(KILL_ENEMY))

        self.x = -self.pos_x + w_screen
        self.render(screen)
