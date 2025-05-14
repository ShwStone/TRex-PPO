import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import random
import pygame

from backgroud import Cloud, Desert, Score
from events import ADD_ENEMY, KILL_ENEMY
from sprites.enemies import Pterodactyl, Cactus
from sprites.player import TRex, TRexStatus
from sprites.collision import detect_collision_by_alpha_channel
from speed import Speed, SpeedRatio
from gameover import GameOver
from PIL import Image
import numpy as np

class TRexRunner :
    def __init__(self, frames=4):
        self.assets_paths = {
            "desert": "./images/desert.png",
            "cloud": "./images/cloud.png",
            "cactuses": [
                "./images/cactus/cactus_1.png",
                "./images/cactus/cactus_2.png",
                "./images/cactus/cactus_3.png",
                "./images/cactus/cactus_4.png",
                "./images/cactus/cactus_5.png",
                "./images/cactus/cactus_6.png",
                "./images/cactus/cactus_7.png",
            ],
            "pterodactyl": [
                "./images/pterodactyl/pterodactyl_1.png",
                "./images/pterodactyl/pterodactyl_2.png",
            ],
            "dinosaur": [
                "./images/t-rex/standing_1.png",
                "./images/t-rex/standing_2.png",
                "./images/t-rex/creeping_1.png",
                "./images/t-rex/creeping_2.png",
            ],
            "gameover": "./images/gameover.png",
            "restart": "./images/restart.png",
            "font": "./fonts/DinkieBitmap-7pxDemo.ttf",
        }

        pygame.init()
        pygame.display.set_caption("T-Rex Runner Pygame")
        self.screen_size = (1000, 350)
        self.down_shape = (500, 175)
        # fps = 5
        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill((255, 255, 255))
        # self.clock = pygame.time.Clock()

        # score = 0.0
        self.speed_ratio = SpeedRatio()
        self.background_speed = Speed(8.0, self.speed_ratio)
        self.cloud_speed = Speed(1.0, self.speed_ratio)

        self.desert_image = pygame.image.load(self.assets_paths["desert"])
        self.cloud_image = pygame.image.load(self.assets_paths["cloud"])
        self.cactus_images = [pygame.image.load(img_path) for img_path in self.assets_paths["cactuses"]]
        self.pterodactyl_images = [pygame.image.load(img_path) for img_path in self.assets_paths["pterodactyl"]]
        self.dinosaur_images = [pygame.image.load(img_path) for img_path in self.assets_paths["dinosaur"]]
        self.gameover_image = pygame.image.load(self.assets_paths["gameover"])
        self.restart_image = pygame.image.load(self.assets_paths["restart"])

        self.desert = Desert(self.desert_image, speed=self.background_speed)
        self.clouds = [
            Cloud(self.cloud_image, 10, 50, speed=self.cloud_speed),
            Cloud(self.cloud_image, 500, 70, speed=self.cloud_speed),
        ]
        self.t_rex = TRex(self.dinosaur_images)
        self.enemies = pygame.sprite.Group()
        self.enemies.add(Cactus(self.cactus_images[0], speed=self.background_speed))
        self.score = Score(font=pygame.font.Font(self.assets_paths["font"], 30), speed=self.background_speed)
        self.gameover = GameOver(self.gameover_image, self.restart_image)
        self.frames = frames
        self.over = False

        self.record_frames = []
        self.clock = pygame.time.Clock()

    def begin(self):
        self.cache = []
        return self.get_frame(0)
    
    def step(self, action, record=False, record_path=None):
        self.kill_an_enemy = False
        state = self.get_frame(action, record, record_path)
        reward = (
            -10 if self.gameover.is_gameover else 
            1 if self.kill_an_enemy else 
            0.05 if self.t_rex.status == TRexStatus.RUNNING else 
            0
        )
        done = self.gameover.is_gameover
        return state, reward, done

    def get_frame(self, action, record=False, record_path=None):
        state = self._step(action)
        if record :
            self.record_frames.append(state)
        
        state = state.resize(self.down_shape)
        state = (np.array(state) < 128)
        # print(state.shape)

        if len(self.cache) == 0 :
            self.cache = [state] * self.frames
        else :
            self.cache.pop(0)
            self.cache.append(state)

        if record and self.gameover.is_gameover :
                self.record_frames[0].save(record_path, save_all=True, append_images=self.record_frames[1:], duration=20, loop=0)
                self.record_frames = []

        return np.array(self.cache)

    def _step(self, action) :
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.over = True
            elif event.type == pygame.QUIT:        
                self.over = True
            elif event.type == ADD_ENEMY:
                if random.random() < 0.4:
                    self.enemies.add(Pterodactyl(self.pterodactyl_images, speed=self.background_speed))
                else:
                    cactus_image = random.choice(self.cactus_images)
                    self.enemies.add(Cactus(cactus_image, speed=self.background_speed))
            elif event.type == KILL_ENEMY:
                self.kill_an_enemy = True

        if self.gameover.is_gameover:
            # 开始新游戏
            self.gameover.is_gameover = False

            for enemy in self.enemies:
                enemy.kill()
            self.enemies.add(Cactus(self.cactus_images[0], speed=self.background_speed))
            self.score.clear()
            self.speed_ratio.reset()
        else:
            self.speed_ratio.update()

            self.desert.update(self.screen)
            for enemy in self.enemies:
                enemy.update(self.screen)
            for cloud in self.clouds:
                cloud.update(self.screen)
            self.t_rex.update(self.screen)
            self.score.update(self.screen)

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE] or pressed[pygame.K_UP]:
                action = 1
            elif pressed[pygame.K_DOWN]:
                action = 2

            self.t_rex.handle_events(action)

            for enemy in self.enemies:
                if detect_collision_by_alpha_channel(self.t_rex, enemy, self.screen, plot_mask=False):
                    self.gameover.is_gameover = True
                    break

        pygame.display.flip()
        # self.clock.tick(60)

        state = pygame.image.tobytes(self.screen, 'RGB')
        state = Image.frombytes('RGB', self.screen.get_size(), state).convert('L')
        return state
    
    def play(self):
        while not self.over:
            self._step(0)
            self.clock.tick(60)
    
    def close(self):
        pygame.quit()
    
    def shape(self):
        return (1, self.frames) + self.down_shape[::-1]

if __name__ == "__main__":
    env = TRexRunner()
    state = env.begin()
    while True:
        state, reward, done = env.step(random.randint(0, 2))
        # frames = [Image.fromarray(state[i]) for i in range(state.shape[0])]
        # frames[0].save(f'output_{time.time()}.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(reward, done)
        # time.sleep(0.01)
        
        if env.over:
            env.close()
            break
    # env.play()