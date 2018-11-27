import pygame, random
import collections
import numpy as np
import os
import cv2
import sys

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from pygame.locals import *
from CollectDataDQN.Screen import Screen

GRID_SIZE = 100

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
FONT_STYLE = pygame.font.match_font('arial')

x = 1
y = 40
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

class SnakeGame():

    def __init__(self):
	    self.reset()

	# Generate a random position to the apple
    def on_grid_random(self):
        x = random.randint(0,GRID_SIZE - 10)
        y = random.randint(0,GRID_SIZE - 10)
        return (x//10 * 10, y//10 * 10)

	# Verifies collision
    def collision(self, c1, c2):
        return (c1[0] == c2[0]) and (c1[1] == c2[1])

	# Draw a text in the screen
    def draw_text(self, surf, text, size, x, y):
        font = pygame.font.Font(FONT_STYLE, size)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def preprocess_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

	# Main function (Run the game for one frame)
    def step(self, action):

        self.clock.tick(10)
		
        end_of_game = False
        reward = -0.01
        
		## Execute action
        if action == 0 and self.direction != DOWN: # UP action
            self.direction = UP
        if action == 2 and self.direction != UP: # DOWN action
            self.direction = DOWN
        if action == 3 and self.direction != RIGHT: # LEFT action
            self.direction = LEFT
        if action == 1 and self.direction != LEFT: # RIGHT action
            self.direction = RIGHT
		
		# Update snake position
        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = (self.snake[i-1][0], self.snake[i-1][1])

        if self.direction == UP:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] - 10)
        if self.direction == DOWN:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] + 10)
        if self.direction == RIGHT:
            self.snake[0] = (self.snake[0][0] + 10, self.snake[0][1])
        if self.direction == LEFT:
            self.snake[0] = (self.snake[0][0] - 10, self.snake[0][1])

		# Update screen elements
        self.screenPyGame.fill((0,0,0))
        self.screenPyGame.blit(self.apple, self.apple_pos)
        i = 0
        for pos in self.snake:
            if i == 0:
                self.screenPyGame.blit(self.head_skin,pos)
                i += 1
            else:
                self.screenPyGame.blit(self.snake_skin,pos)

		# Draw Score
        #self.draw_text(self.screenPyGame, ('Score: ' + str(self.score)), 18, (GRID_SIZE + 200) / 2, 10)

        pygame.display.update()
		
		# Verifies collision with the apple
        if self.collision(self.snake[0], self.apple_pos):
            self.apple_pos = self.on_grid_random()
            self.snake.append((0,0))
            self.score += 10
            reward = 1
            #end_of_game = True

	    # Collisions with walls or Collision with the another part of the snake
        if (self.snake[0][0] < 0 or self.snake[0][1] < 0 or self.snake[0][0] > GRID_SIZE - 10 or self.snake[0][1] > GRID_SIZE - 10) or (len(self.snake) != len(set(self.snake))):
            #print('GAME OVER - Total Score = ', self.score)
            end_of_game = True
            reward = -1
		
        self.frames.append(self.preprocess_image(self.screen_state()))

        #return self.get_frames(), reward, end_of_game # for dqn 4 frames
        return self.screen_state(), reward, end_of_game, self.score # for rgb
				
    def get_frames(self):
        return np.array(list(self.frames))

    def screen_state(self):
        return self.screen.GrabScreenBGR()
		
    def quit_game(self):
        pygame.quit()
		
    def reset(self): 
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption('Snake')
		
        self.frames = collections.deque(maxlen=4)
		
		# Initialize screen
        self.screenPyGame = pygame.display.set_mode((GRID_SIZE,GRID_SIZE))
        self.screen = Screen(region=(0,40,GRID_SIZE,GRID_SIZE+40))
		
	    # Initialize score
        self.score = 0
		
		# Initialize direction
        self.direction = LEFT
		
		# Initialize snake
        self.snake = [(50, 50), (60, 50), (70,50)]
        self.head_skin = pygame.Surface((10,10))
        self.head_skin.fill((25,255,25))
        self.snake_skin = pygame.Surface((10,10))
        self.snake_skin.fill((255,255,255))
        
		# Initialize apple
        self.apple_pos = self.on_grid_random()
        self.apple = pygame.Surface((10,10))
        self.apple.fill((255,0,0))
		
		# Initialize clock
        self.clock = pygame.time.Clock()

        #return self.preprocess_image(self.screen_state()) # for grayscale
        #return self.screen_state() # for RGB
		
if __name__ == '__main__':
	
    snake_game = SnakeGame()