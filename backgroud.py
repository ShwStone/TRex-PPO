import math
import json
import os

import pygame

from speed import Speed


class Desert:
    def __init__(self, image: pygame.Surface, speed: Speed) -> None:
        self.image = image
        self.speed = speed
        self.pos_x = 0.0

    def update(self, screen: pygame.Surface) -> None:
        w_screen, h_screen = screen.get_size()
        w_image, h_image = self.image.get_size()
        self.pos_x -= self.speed.value
        self.pos_x %= w_image

        # 计算当前屏幕需要几张背景才能全覆盖
        n_images = math.ceil(w_screen / w_image) + 1
        screen.fill((255, 255, 255))
        for i in range(n_images):
            x = self.pos_x % w_image + w_image * (i - 1)
            y = h_screen - h_image
            screen.blit(self.image, (x, y))


class Cloud:
    def __init__(self, image: pygame.Surface, x: int, y: int, speed: Speed) -> None:
        self.image = image
        self.speed = speed
        self.x = x
        self.y = y

    def update(self, screen: pygame.Surface) -> None:
        w_screen, h_screen = screen.get_size()
        w_image, h_image = self.image.get_size()
        self.x -= self.speed.value
        self.x %= w_screen + w_image
        screen.blit(self.image, (self.x - w_image, self.y))

import atexit

class Score:
    dump_file = "./score.json"

    def __init__(self, font: pygame.font.Font, speed: Speed) -> None:
        self.font = font
        self.speed = speed
        self.value = 0.0
        self.highest_value = 0.0
        if os.path.isfile(Score.dump_file):
            with open(Score.dump_file, "r", encoding="utf-8") as f:
                self.highest_value = json.load(f)["highest_score"]

    def __del__(self):
        atexit.register(self.record)

    def record(self):
        with open(Score.dump_file, "w", encoding="utf-8") as f:
            json.dump({"highest_score": self.highest_value}, f, indent=2)

    def update(self, screen: pygame.Surface) -> None:
        w_screen, h_screen = screen.get_size()
        self.value += self.speed.value * 0.01
        if self.value > self.highest_value:
            self.highest_value = self.value

        text = self.font.render(f"HI {int(self.highest_value):05} {int(self.value):05}", True, (83, 83, 83))
        screen.blit(text, (w_screen - 10 - text.get_size()[0], 30))

    def clear(self) -> None:
        self.value = 0.0
