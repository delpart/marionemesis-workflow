import pygame as pg
import numpy as np

class Tile(pg.sprite.Sprite):
    def __init__(self, pos, size, img):
        super(Tile, self).__init__()
        self.image = pg.Surface((size, size), pg.SRCALPHA, 32)
        # pg.pixelcopy.array_to_surface(self.image, img[..., :3])
        # alpha = np.array(self.image.get_view('A'), copy=False)
        # alpha = img[..., 3]
        # del alpha
        for row in range(self.image.get_height()):
            for col in range(self.image.get_width()):
                self.image.set_at((row, col), img[row, col])
        self.image.convert_alpha()
        self.rect = self.image.get_rect(topleft=pos)

    def update(self, shift):
        self.rect.x += shift.x
        self.rect.y += shift.y
