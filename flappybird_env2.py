"""Implementation based on
https://github.com/techwithtim/NEAT-Flappy-Bird/blob/master/flappy_bird.py"""

import os
import random

import pyglet
###pyglet.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
###STAT_FONT = pygame.font.SysFont("comicsans", 50)
###END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False





class Env:
    window = pyglet.window.Window(width=WIN_WIDTH, height=WIN_HEIGHT,
            caption="Flappy Bird")
    pipe_img = pyglet.sprite.Sprite(pyglet.image.load(
        os.path.join("flappybird_imgs", "pipe.png"))).update(scale=2.0)
    bg_img = pyglet.sprite.Sprite(pyglet.image.load(os.path.join("flappybird_imgs", "bg.png")))
    bg_img.width = 600
    bg_img.height = 800
    bird_images = [pyglet.sprite.Sprite(pyglet.image.load(os.path.join("flappybird_imgs", "bird" + str(x) + ".png"))) for x in range(1, 4)]
    base_img = pyglet.sprite.Sprite(pyglet.image.load(
        os.path.join("flappybird_imgs", "base.png")))

    pyglet.app.run()

    def __init__(self):
        self.bird = Bird(230, 350)
        self.base = Base(FLOOR)
        self.pipes = [Pipe(700)]
        self.gen = 0
        self.score = 0

        self.state = (self.bird.y, abs(self.bird.y - self.pipes[0].height),
                      abs(self.bird.y - self.pipes[0].bottom))
        self.reward = 0.0
        self.done = False

    @window.event
    def on_draw():
        Env.window.clear()
        #Env.pipe_img.draw()
        Env.bg_img.draw()
        Env.base_img.draw()
        #self.bird.draw_window()


    def step(self, action):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                    self.done = True

            pipe_ind = 0
            if len(self.pipes) > 1 and self.bird.x > self.pipes[0].x +
                                        self.pipes[0].PIPE_TOP.width:
                pipe_ind = 1

            self.reward += 0.1
            self.bird.move()
            if action:
                self.bird.jump()

            self.base.move()

            rem = []
            add_pipe = False
            for pipe in self.pipes:
                pipe.move()
                # check for collision
                if pipe.collide(self.bird):
                    self.reward -= 1.0
                    self.done = True

                if pipe.x + pipe.PIPE_TOP.width < 0:
                    rem.append(pipe)

                if not pipe.passed and pipe.x < self.bird.x:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                self.score += 1
                self.reward += 5.0
                self.pipes.append(Pipe(WIN_WIDTH))

            for r in rem:
                self.pipes.remove(r)

            if self.bird.y + self.bird.img.height - 10 >= FLOOR or self.bird.y < -50:
                self.reward -= 1
                self.done = True

            state = (self.bird.y, abs(self.bird.y - self.pipes[pipe_ind].height),
                     abs(self.bird.y - self.pipes[pipe_ind].bottom))
            reward = self.reward
            self.reward = 0

            draw_window(Env.WIN, self.bird, self.pipes, self.base, self.score, self.gen, pipe_ind)

            yield state, reward, self.done




class Bird:
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    IMGS = Env.bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self):
        """
        draw the bird
        :param
        :return: None
        """
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2


        # tilt the bird
        rotated_bird = pyglet.sprite.Sprite(self.image).update(rotation=self.tilt)
        rotated_bird.draw()




class Pipe():
    """
    represents a pipe object
    """
    GAP = 200
    VEL = 5

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = Env.pipe_img.update(rotation=180.0)
        self.PIPE_BOTTOM = Env.pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.height
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        self.PIPE_TOP.position.update(x=self.x, y=self.top).draw()
        # draw bottom
        self.PIPE_BOTTOM.update(x=self.x, y=self.bottom).draw()


    def collide(self, bird, base):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        if self.bird.x <= self.x + self.PIPE_TOP.width + 15 and self.bird.x >= self.x and\
           self.bird.y <= self.y :
            self.bird
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False




class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    WIDTH = Env.base_img.width
    IMG = Env.base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self):
        """
        Draw the floor. This is two images that move together.
        :param
        :return: None
        """
        self.IMG.update(x=self.x1, y=self.y).draw()
        self.IMG.update(x=self.x2, y=self.y).draw()


def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    """
    draws the windows for the main game loop
    :param win: pygame window surface
    :param bird: a Bird object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param gen: current generation
    :param pipe_ind: index of closest pipe
    :return: None
    """
    if gen == 0:
        gen = 1

    Env.bg_img.draw()

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win,
                                 (255, 0, 0),
                                 (bird.x+bird.img.width/2, bird.y + bird.img.height/2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.width/2,
                                  pipes[pipe_ind].height),
                                 5)
                pygame.draw.line(win,
                                 (255, 0, 0),
                                 (bird.x+bird.img.width/2, bird.y + bird.img.height/2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.width/2,
                                  pipes[pipe_ind].bottom),
                                 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # generations
    score_label = STAT_FONT.render("Gens: " + str(gen-1), 1, (255, 255, 255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(score_label, (10, 50))

    pygame.display.update()
