import pygame


class GameOver:
    def __init__(self, gameover_image: pygame.Surface, restart_image: pygame.Surface) -> None:
        self.gameover_image = gameover_image
        self.restart_image = restart_image
        self.is_gameover = False

    def update(self, screen: pygame.Surface) -> None:
        w_screen, h_screen = screen.get_size()
        w_gameover, h_gameover = self.gameover_image.get_size()
        w_restart, h_restart = self.restart_image.get_size()

        screen.blit(self.gameover_image, ((w_screen - w_gameover) / 2, (h_screen - h_gameover) / 2 - 20))
        screen.blit(self.restart_image, ((w_screen - w_restart) / 2, (h_screen - h_restart) / 2 + 50))

    def handle_events(self, pressed_keys: pygame.key.ScancodeWrapper) -> None:
        if self.is_gameover:
            if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_SPACE]:
                self.is_gameover = False
