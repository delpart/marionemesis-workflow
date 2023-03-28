import pygame as pg

from .tiles import Tile
from .assets import Assets
from .player import Player


class Level:
    def __init__(self, surface):
        self.display_surface = surface
        self.assets = Assets()
        self.shift = pg.math.Vector2(0, 0)
        self.camera_center = pg.math.Vector2(surface.get_width()//2, 0)
        with open('assets/levels/map_init.txt', 'r') as f:
            self.map_init = f.readlines()
        self.tiles = pg.sprite.Group()
        self.player = pg.sprite.GroupSingle()
        self.mario_init = pg.math.Vector2(0, 0)
        self.map_rows = [['-']*16]*33

        for i, line in enumerate(self.map_init):
            for j, c in enumerate(line):
                if c == '\n':
                    continue
                self.map_rows[j][i] = c
                if c == 'M':
                    self.player.add(Player((j * 16, i * 16), self.assets.smallmario))
                    self.mario_init = pg.math.Vector2((j * 16, i * 16))
                    self.pos_mario = self.player.sprite.rect.center
                else:
                    img = None
                    if c == 'X':
                        img = self.assets.map[0][1]
                    elif c == '#':
                        img = self.assets.map[0][2]
                    elif c == '<':
                        img = self.assets.map[2][2]
                    elif c == '>':
                        img = self.assets.map[2][3]
                    elif c == '[':
                        img = self.assets.map[2][4]
                    elif c == ']':
                        img = self.assets.map[2][5]
                    if img is not None:
                        self.tiles.add(Tile((j * 16, i * 16), 16, img))

    def append_row(self, row):
        self.map_rows.append(['-']*16)
        for j, c in enumerate(row):
            if c == '\n':
                continue
            self.map_rows[-1][j] = c
            img = None
            if c == 'X':
                img = self.assets.map[0][1]
            elif c == '#':
                img = self.assets.map[0][2]
            elif c == '<':
                img = self.assets.map[2][2]
            elif c == '>':
                img = self.assets.map[2][3]
            elif c == '[':
                img = self.assets.map[2][4]
            elif c == ']':
                img = self.assets.map[2][5]
            if img is not None:
                self.tiles.add(Tile((self.display_surface.get_width() + self.camera_center.x%16 - 32, j * 16), 16, img))
        self.map_rows = self.map_rows[1:]

    def step(self):
        dead = self.update_and_render()

        removed_sprite = False
        for sprite in self.tiles.sprites():
            if sprite.rect.right < 0:
                removed_sprite = True
                self.tiles.remove(sprite)
        # observation, reward, terminated, truncated, info
        return dead, removed_sprite

    def update(self):
        self.camera_center -= self.shift
        self.tiles.update(self.shift)
        self.scroll_x()
        self.player.update()
        self.handle_horizontal_collisions()
        self.handle_vertical_collisions()

        self.pos_mario = int(self.camera_center.x - (self.display_surface.get_width()/2 - self.display_surface.get_width()/3) - (self.display_surface.get_width()/3 - self.player.sprite.rect.centerx)), self.player.sprite.rect.centery

    def handle_horizontal_collisions(self):
        player = self.player.sprite
        player.rect.x += player.direction.x * player.speed

        if player.rect.centerx < 8:
            player.rect.centerx = 8

        for sprite in self.tiles.sprites():
            if sprite.rect.colliderect(player.rect):
                if player.direction.x < 0:
                    player.rect.left = sprite.rect.right
                elif player.direction.x > 0:
                    player.rect.right = sprite.rect.left

    def handle_vertical_collisions(self):
        player = self.player.sprite
        player.direction.y += player.gravity
        player.rect.y += player.direction.y

        for sprite in self.tiles.sprites():
            if sprite.rect.colliderect(player.rect):
                if player.direction.y < 0:
                    player.rect.top = sprite.rect.bottom
                    player.direction.y = 0
                elif player.direction.y > 0:
                    player.direction.y = 0
                    player.rect.bottom = sprite.rect.top

    def check_death(self):
        return self.player.sprite.rect.centery > self.display_surface.get_height()

    def render(self):
        self.display_surface.fill((100, 149, 237))
        self.tiles.draw(self.display_surface)
        self.player.draw(self.display_surface)

    def update_and_render(self):
        self.update()
        self.render()
        return self.check_death()

    def scroll_x(self):
        player = self.player.sprite
        player_x = player.rect.centerx
        direction_x = player.direction.x

        if player_x > self.display_surface.get_width()/3 and direction_x > 0:
            player.speed = 0
            self.shift.x = -8
        else:
            player.speed = 8
            self.shift.x = 0
