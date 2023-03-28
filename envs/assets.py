from PIL import Image
import numpy as np


class Assets:
    def __init__(self):
        self.mario = cutimage('assets/mariosheet.png', 32, 32)
        self.smallmario = cutimage('assets/smallmariosheet.png', 16, 16)
        self.firemario = cutimage('assets/firemariosheet.png', 32, 32)
        self.enemy = cutimage('assets/enemysheet.png', 16, 32)
        self.items = cutimage('assets/itemsheet.png', 16, 16)
        self.map = cutimage('assets/mapsheet.png', 16, 16)
        self.particles = cutimage('assets/particlesheet.png', 16, 16)
        self.font = cutimage('assets/font.gif', 8, 8)


def cutimage(image_path, x_size, y_size):
    source = np.array(Image.open(image_path))
    images = []

    for y in range(source.shape[0]//y_size):
        images.append([])
        for x in range(source.shape[1]//x_size):
            sprite = np.flipud(np.rot90(source[y*y_size:(y+1)*y_size, x*x_size:(x+1)*x_size]))
            images[-1].append(sprite)
    return images
