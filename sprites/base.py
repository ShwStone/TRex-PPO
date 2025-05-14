import pygame

import numpy as np


class SpriteBase(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()

    def update(self, screen: pygame.Surface) -> None:
        raise NotImplementedError

    def render(self, screen: pygame.Surface) -> None:
        screen.blit(self.image, (self.x, self.y))

    def get_screen_mask(self, screen_size: tuple[int, int]) -> np.ndarray:
        screen_tp = np.zeros(screen_size, dtype=np.uint8)
        w_screen, h_screen = screen_size
        w, h = self.image.get_size()
        img_arr = pygame.surfarray.array_alpha(self.image)
        x1, x2 = int(self.x), int(self.x) + w
        y1, y2 = int(self.y), int(self.y) + h
        if x2 <= 0 or x1 > w_screen or y2 <= 0 or y1 > h_screen:
            return screen_tp

        screen_tp[max(x1, 0) : min(x2, w_screen), max(y1, 0) : min(y2, h_screen)] = img_arr[
            max(0, -x1) : w - max(0, x2 - w_screen), max(0, -y1) : h - max(0, y2 - h_screen)
        ]
        return screen_tp
