"""
 Pygame base template for opening a window

 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/

 Explanation video: http://youtu.be/vRB_983kUMc

-------------------------------------------------

Author for the Brickout game is Christian Bender
That includes the classes Ball, Paddle, Brick, and BrickWall.

"""

import random

# using pygame python GUI
import pygame

# Define Four Colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.init()

# Setting the width and height of the screen [width, height]
size = (700, 500)
screen = pygame.display.set_mode(size)


class Ball(object):
    """
    A class representing a ball in the Brickout game.

    Attributes:
        __screen: The game screen surface
        _radius: Radius of the ball
        _xLoc: X-coordinate of the ball's center
        _yLoc: Y-coordinate of the ball's center
        __xVel: X-axis velocity of the ball
        __yVel: Y-axis velocity of the ball
        __width: Width of the game screen
        __height: Height of the game screen
    """

    def __init__(self, screen, radius, x, y):
        self.__screen = screen
        self._radius = radius
        self._xLoc = x
        self._yLoc = y
        self.__xVel = 7
        self.__yVel = 2
        w, h = pygame.display.get_surface().get_size()
        self.__width = w
        self.__height = h

    def getXVel(self):
        """Get the X velocity of the ball."""
        return self.__xVel

    def getYVel(self):
        """Get the Y velocity of the ball."""
        return self.__yVel

    def draw(self):
        """
        Draw the ball onto the screen.
        :param self: self
        :return: None
        """
        pygame.draw.circle(screen, (255, 0, 0), (self._xLoc, self._yLoc), self._radius)

    def update(self, paddle, brickwall):
        """
         Update the ball's position and handle collisions.
        :param self: self
        :param paddle: The paddle object for collision detection
        :param brickwall: The brick wall object for collision detection
        :return: True if ball dropped out of bottom, else return false
        """
        self._xLoc += self.__xVel
        self._yLoc += self.__yVel
        # left screen wall bounce
        if self._xLoc <= self._radius:
            self.__xVel *= -1
        # right screen wall bounce
        elif self._xLoc >= self.__width - self._radius:
            self.__xVel *= -1
        # top wall bounce
        if self._yLoc <= self._radius:
            self.__yVel *= -1
        # bottom drop out
        elif self._yLoc >= self.__width - self._radius:
            return True

        # for bouncing off the bricks.
        if brickwall.collide(self):
            self.__yVel *= -1

        # collision detection between ball and paddle
        paddleY = paddle._yLoc
        paddleW = paddle._width
        paddleH = paddle._height
        paddleX = paddle._xLoc
        ballX = self._xLoc
        ballY = self._yLoc

        if ((ballX + self._radius) >= paddleX and ballX <= (paddleX + paddleW)) and (
            (ballY + self._radius) >= paddleY and ballY <= (paddleY + paddleH)
        ):
            self.__yVel *= -1

        return False


class Paddle(object):
    """
    A class representing the player's paddle in the Brickout game.

    Attributes:
        __screen: The game screen surface
        _width: Width of the paddle
        _height: Height of the paddle
        _xLoc: X-coordinate of the paddle's top-left corner
        _yLoc: Y-coordinate of the paddle's top-left corner
        __W: Width of the game screen
        __H: Height of the game screen
    """

    def __init__(self, screen, width, height, x, y):
        self.__screen = screen
        self._width = width
        self._height = height
        self._xLoc = x
        self._yLoc = y
        w, h = pygame.display.get_surface().get_size()
        self.__W = w
        self.__H = h

    def draw(self):
        """
        Draw the paddle onto the screen.
        :return: None
        """
        pygame.draw.rect(
            screen, (0, 0, 0), (self._xLoc, self._yLoc, self._width, self._height), 0
        )

    def update(self):
        """
        moves the paddle at the screen via mouse
        :return: None
        """
        x, y = pygame.mouse.get_pos()
        if x >= 0 and x <= (self.__W - self._width):
            self._xLoc = x


class Brick(pygame.sprite.Sprite):
    """
    A class representing an individual brick in the game.

    Attributes:
        __screen: The game screen surface
        _width: Width of the brick
        _height: Height of the brick
        _xLoc: X-coordinate of the brick's top-left corner
        _yLoc: Y-coordinate of the brick's top-left corner
        __W: Width of the game screen
        __H: Height of the game screen
        __isInGroup: Boolean indicating if brick is part of a group
    """

    def __init__(self, screen, width, height, x, y):
        self.__screen = screen
        self._width = width
        self._height = height
        self._xLoc = x
        self._yLoc = y
        w, h = pygame.display.get_surface().get_size()
        self.__W = w
        self.__H = h
        self.__isInGroup = False

    def draw(self):
        """
        Draw the brick onto the screen with blue color.
        color: rgb(56, 177, 237)
        :return: None
        """
        pygame.draw.rect(
            screen,
            (56, 177, 237),
            (self._xLoc, self._yLoc, self._width, self._height),
            0,
        )

    def add(self, group):
        """
        Add this brick to a given group.
        :param group: The group to add the brick to
        :return: None
        """
        group.add(self)
        self.__isInGroup = True

    def remove(self, group):
        """
        Remove this brick from the given group.
        :param group: The group to remove the brick from
        :return: None
        """
        group.remove(self)
        self.__isInGroup = False

    def alive(self):
        """
        Check if the brick belongs to the brick wall.
        :return: True if brick is in a group, else return false
        """
        return self.__isInGroup

    def collide(self, ball):
        """
         Check collision between the ball and this brick.
        :param ball: The ball object to check collision with
        :return: True if collision detected, else return false
        """
        brickX = self._xLoc
        brickY = self._yLoc
        brickW = self._width
        brickH = self._height
        ballX = ball._xLoc
        ballY = ball._yLoc

        if (
            (ballX + ball._radius) >= brickX
            and (ballX + ball._radius) <= (brickX + brickW)
        ) and (
            (ballY - ball._radius) >= brickY
            and (ballY - ball._radius) <= (brickY + brickH)
        ):
            return True
        else:
            return False


class BrickWall(pygame.sprite.Group):
    """
    A class representing a wall of bricks in the game.

    Attributes:
        __screen: The game screen surface
        _x: Starting X-coordinate of the brick wall
        _y: Starting Y-coordinate of the brick wall
        _width: Width of each brick
        _height: Height of each brick
        _bricks: List containing all bricks in the wall
    """

    def __init__(self, screen, x, y, width, height):
        self.__screen = screen
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._bricks = []

        X = x
        Y = y
        for i in range(3):
            for j in range(4):
                self._bricks.append(Brick(screen, width, height, X, Y))
                X += width + (width / 7.0)
            Y += height + (height / 7.0)
            X = x

    def add(self, brick):
        """
        Add a brick to this BrickWall.
        :param brick: The brick to add to the wall
        :return: None
        """
        self._bricks.append(brick)

    def remove(self, brick):
        """
        Remove a brick from this BrickWall.
        :param brick: The brick to remove from the wall
        :return: None
        """
        self._bricks.remove(brick)

    def draw(self):
        """
        Draw all bricks in the wall onto the screen.
        :return: None
        """
        for brick in self._bricks:
            if brick is not None:
                brick.draw()

    def update(self, ball):
        """
        Check and handle collisions between ball and bricks.
        :param ball:The ball object to check collisions with
        :return:
        """
        for i in range(len(self._bricks)):
            if (self._bricks[i] is not None) and self._bricks[i].collide(ball):
                self._bricks[i] = None

        # removes the None-elements from the brick list.
        for brick in self._bricks:
            if brick is None:
                self._bricks.remove(brick)

    def hasWin(self):
        """
        Check if player has won the game.
        :return:True if all bricks are destroyed, else return false
        """
        return len(self._bricks) == 0

    def collide(self, ball):
        """
         Check collisions between the ball and any brick in the wall.
        :param ball:The ball object to check collisions with
        :return:True if collision detected with any brick, else return false
        """
        for brick in self._bricks:
            if brick.collide(ball):
                return True
        return False


# The game objects ball, paddle and brick wall
ball = Ball(screen, 25, random.randint(1, 700), 250)
paddle = Paddle(screen, 100, 20, 250, 450)
brickWall = BrickWall(screen, 25, 25, 150, 50)

isGameOver = False  # determines whether game is lose
gameStatus = True  # game is still running

score = 0  # score for the game.

pygame.display.set_caption("Brickout-game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# for displaying text in the game
pygame.font.init()  # you have to call this at the start,
# if you want to use this module.

# message for game over
mgGameOver = pygame.font.SysFont("Comic Sans MS", 40)

# message for winning the game.
mgWin = pygame.font.SysFont("Comic Sans MS", 40)

# message for score
mgScore = pygame.font.SysFont("Comic Sans MS", 40)

textsurfaceGameOver = mgGameOver.render("Game Over!", False, (0, 0, 0))
textsurfaceWin = mgWin.render("You win!", False, (0, 0, 0))
textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # --- Game logic should go here

    # --- Screen-clearing code goes here

    # Here, we clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.

    # If you want a background image, replace this clear with blit'ing the
    # background image.
    screen.fill(WHITE)

    # --- Drawing code should go here

    """
        Because I use OOP in the game logic and the drawing code,
        are both in the same section.
    """
    if gameStatus:
        # first draws ball for appropriate displaying the score.
        brickWall.draw()

        # for counting and displaying the score
        if brickWall.collide(ball):
            score += 10
        textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
        screen.blit(textsurfaceScore, (300, 0))

        # after scoring. because hit bricks are removed in the update-method
        brickWall.update(ball)

        paddle.draw()
        paddle.update()

        if ball.update(paddle, brickWall):
            isGameOver = True
            gameStatus = False

        if brickWall.hasWin():
            gameStatus = False

        ball.draw()

    else:  # game isn't running.
        if isGameOver:  # player lose
            screen.blit(textsurfaceGameOver, (0, 0))
            textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
            screen.blit(textsurfaceScore, (300, 0))
        elif brickWall.hasWin():  # player win
            screen.blit(textsurfaceWin, (0, 0))
            textsurfaceScore = mgScore.render("score: " + str(score), False, (0, 0, 0))
            screen.blit(textsurfaceScore, (300, 0))

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
pygame.quit()
