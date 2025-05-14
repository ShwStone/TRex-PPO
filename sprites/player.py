from enum import Enum

import pygame

from sprites.base import SpriteBase


class TRex(SpriteBase):
    def __init__(self, images: list[pygame.Surface]) -> None:
        super().__init__()
        self.images = images
        self.index = 0
        self.jumping_index = 0
        self.num_shift = 6
        self.status = TRexStatus.RUNNING

    @property
    def image(self) -> pygame.Surface:
        img: pygame.Surface
        if self.status == TRexStatus.CREEPING:
            img = self.images[2 + self.index // self.num_shift]
        else:
            img = self.images[self.index // self.num_shift]
        return img

    @staticmethod
    def jumping_height(t: float) -> float:
        t *= 0.01
        return 1500 * (t - 3 * t ** 2)

    def handle_events(self, action) -> None:
        if self.status != TRexStatus.JUMPING:
            self.status = TRexStatus.RUNNING
            if action == 1:
                self.status = TRexStatus.JUMPING
            elif action == 2:
                self.status = TRexStatus.CREEPING

    def update(self, screen: pygame.Surface) -> None:
        w_screen, h_screen = screen.get_size()
        w_image, h_image = self.image.get_size()

        if self.status != TRexStatus.JUMPING:
            self.index += 1
            self.index %= self.num_shift * 2
        else:
            self.jumping_index += 1
            if TRex.jumping_height(self.jumping_index) < 0:
                self.jumping_index = 0
                self.status = TRexStatus.RUNNING

        self.x = 20
        if self.status == TRexStatus.JUMPING:
            self.y = h_screen - h_image - TRex.jumping_height(self.jumping_index)
        else:
            self.y = h_screen - h_image
        self.render(screen)


class TRexStatus(Enum):
    RUNNING = 0
    JUMPING = 1
    CREEPING = 2
