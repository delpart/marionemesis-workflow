from typing import Optional, Union, List

import numpy as np
import pygame as pg
import cv2

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame

from .assets import Assets
from .level import Level


char_to_int = {
    '-': 0,
    'X': 1,
    '#': 2,
    '<': 3,
    '>': 4,
    '[': 5,
    ']': 6
}

int_to_char = {v: k for k, v in char_to_int.items()}

class MarioNemesis(gym.Env):
    def __init__(self):
        self.assets = Assets()
        self.lifes = 3
        self.observation_space = spaces.Dict({
            'nemesis': spaces.Dict({
                'level': spaces.Box(0, 7, shape=(512//16+1, 256//16), dtype=int),
                'pos_mario': spaces.Box(0, 256, shape=(2,), dtype=int),
                'elapsed_steps': spaces.Box(0, np.inf, shape=(1,), dtype=int)
            }),
            'mario': spaces.Box(0, 255, shape=(512, 256, 3))
        })

        pg.init()
        self.display = pg.display.set_mode((512, 256))
        self.level = None
        self.mario_steps = 0
        self.mario_jumps = 0
        self.mario_init = 0

    def render(self):
        return pg.surfarray.array3d(self.display)

    def step(self, mario_actions, nemesis_actions=None):
        nemesis_penalty = 0
        if nemesis_actions is not None:
            nemesis_actions = nemesis_actions.reshape(16, 7)
            nemesis_actions_str = ''
            for i in range(nemesis_actions.shape[0]):
                nemesis_actions_str += int_to_char[np.argmax(nemesis_actions[i]).astype(int)]
                if int_to_char[np.argmax(nemesis_actions[i]).astype(int)] != '-':
                    nemesis_penalty += 1
            self.level.append_row(nemesis_actions_str)
        current_pos = self.level.pos_mario
        key_dict = {pg.K_LEFT: mario_actions[0] > 0, pg.K_RIGHT: mario_actions[1] > 0, pg.K_SPACE: mario_actions[2] > 0}

        self.level.player.sprite.external_input = key_dict

        dead, step_nemesis = self.level.step()
        self.mario_steps += 1
        self.mario_steps += 1
        self.mario_jumps += 1*(mario_actions[2] > 0)
        pg.display.update()

        elapsed_steps = self.mario_steps
        num_jumps = self.mario_jumps
        if step_nemesis:
            self.mario_jumps = 0
            self.mario_steps = 0

        level_array = np.zeros((33, 16), dtype=int)

        for i, col in enumerate(self.level.map_rows):
            for j, c in enumerate(col):
                level_array[i, j] = char_to_int[c]
        stuck = False
        if self.mario_steps > 100:
            dead = True
            stuck = True

        reward_nemesis = np.array([self.level.pos_mario[0] - current_pos[0] - 10*dead - 50*stuck + num_jumps - nemesis_penalty], dtype=np.float32)

        walked_distance = 0
        if dead or stuck:
            walked_distance = self.mario_init - current_pos[0]
            if walked_distance <= 0:
                walked_distance = -100
        action_taken = np.sum(mario_actions) > 0
        reward_mario = np.array([walked_distance + self.level.pos_mario[0] - current_pos[0] - 50*dead + action_taken*0.5 + 0.5*(mario_actions[2] > 0)], dtype=np.float32)
        # observation, reward, terminated, truncated, info

        return {'mario': self._get_mario_obs(), 'nemesis': {'elapsed': np.array([elapsed_steps]), 'pos_mario': np.array(self.level.pos_mario), 'level': level_array}}, {'nemesis': reward_nemesis, 'mario': reward_mario}, dead, None, {'step_nemesis': step_nemesis}

    def _get_mario_obs(self):
        img = pg.surfarray.array3d(self.display).astype(np.float32)/255.
        img = 0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]
        img = cv2.resize(img, dsize=(128, 256), interpolation=cv2.INTER_CUBIC)
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = Level(self.display)
        self.level.render()
        self.mario_jumps = 0
        self.mario_steps = 0
        self.mario_init = self.level.pos_mario[0]
        level_array = np.zeros((33, 16), dtype=int)

        for i, col in enumerate(self.level.map_rows):
            for j, c in enumerate(col):
                level_array[i, j] = char_to_int[c]
        return {'mario': self._get_mario_obs(), 'nemesis': {'elapsed': np.array([0]), 'pos_mario': np.array(self.level.pos_mario), 'level': level_array}}, {'step_nemesis': False}

    def run(self):
        import sys
        pg.init()

        self.display = pg.display.set_mode((512, 256))
        clock = pg.time.Clock()

        for _ in range(3):
            self.level = Level(self.display)
            self.level.player.sprite.human_control = True
            while True:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        sys.exit()

                dead, add_row = self.level.step()

                if dead:
                    break

                if add_row:
                    self.level.append_row('-'*14 + 'XX')

                pg.display.update()
                clock.tick(30)
