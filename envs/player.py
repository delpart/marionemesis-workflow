import pygame as pg


class Player(pg.sprite.Sprite):
    def __init__(self, pos, sprite_sheet):
        super(Player, self).__init__()

        self.animations = {'idle': [], 'run': [], 'jump': [], 'fall': []}

        for i, img in enumerate(sprite_sheet[0]):
            surf = pg.Surface((img.shape[0], img.shape[1]), pg.SRCALPHA, 32)
            for row in range(surf.get_height()):
                for col in range(surf.get_width()):
                    surf.set_at((row, col), img[row, col])
            surf.convert_alpha()

            if i == 0:
                self.animations['idle'].append(surf)
            elif 0 < i < 4:
                self.animations['run'].append(surf)
            elif 4 <= i < 6:
                self.animations['jump'].append(surf)
            elif i == 6:
                self.animations['fall'].append(surf)

        self.image = self.animations['idle'][0]
        self.rect = self.image.get_rect(topleft=pos)

        self.frame_index = 0
        self.animation_speed = 0.15

        self.direction = pg.math.Vector2(0,0)
        self.orientation = 0
        self.speed = 8
        self.gravity = 0.8
        self.jump_speed = -11
        self.state = 'idle'
        self.last_state = 'idle'
        self.external_input = {pg.K_LEFT: False, pg.K_RIGHT: False, pg.K_SPACE: False}
        self.human_control = False

    def update(self):
        self.get_input()
        self.update_state()
        self.animate()

    def update_state(self):
        if self.direction.y < 0:
            self.state = 'jump'
        elif self.last_state == 'jump' and self.direction.y > 0:
            self.state = 'fall'
        elif self.direction.y > self.gravity:
            self.state = 'fall'
        else:
            if self.direction.x != 0:
                self.state = 'run'
            else:
                self.state = 'idle'
        self.last_state = self.state

        if self.direction.x < 0:
            self.orientation = -1
        elif self.direction.x > 0:
            self.orientation = 1

    def animate(self):
        self.frame_index += self.animation_speed

        self.image = self.animations[self.state][int(self.frame_index)%len(self.animations[self.state])]

        if self.orientation < 0:
            self.image = pg.transform.flip(self.image, True, False)

    def get_input(self):
        if self.human_control:
            keys = pg.key.get_pressed()
        else:
            keys = self.external_input

        if keys[pg.K_RIGHT]:
            self.direction.x = 1
        elif keys[pg.K_LEFT]:
            self.direction.x = -1
        else:
            self.direction.x = 0

        if (keys[pg.K_SPACE] or keys[pg.K_UP]) and self.state not in ['jump', 'fall']:
            self.direction.y = self.jump_speed
