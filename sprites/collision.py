import numpy as np
import pygame

import matplotlib.pyplot as plt
from sprites.base import SpriteBase


def detect_collision_by_alpha_channel(a: SpriteBase, b: SpriteBase, screen: pygame.Surface, plot_mask: bool = False):
    screen_size = screen.get_size()
    mask_a = a.get_screen_mask(screen_size)
    mask_b = b.get_screen_mask(screen_size)

    if plot_mask:
        plt.imshow(np.transpose(mask_a & mask_b), cmap="gray")
        plt.pause(0.0000001)
        plt.clf()

    if (mask_a & mask_b).any():
        return True
    return False


def detect_collision_by_border(a: SpriteBase, b: SpriteBase):
    w_a, h_a = a.image.get_size()
    w_b, h_b = b.image.get_size()
    x1_l, x1_r = a.x, a.x + w_a
    x2_l, x2_r = b.x, b.x + w_b
    y1_u, y1_d = a.y, a.y + h_a
    y2_u, y2_d = b.y, b.y + h_b

    if (x1_l < x2_r and x2_l < x1_r) and (y1_u < y2_d and y2_u < y1_d):
        return True
    return False
