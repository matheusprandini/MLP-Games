import numpy as np
import cv2
import time
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], ''))

from CollectDataHuman.Screen import Screen
from CollectDataHuman.Keys import Keys, W, A, S, D, N, SPACE

class Agent():

    def __init__(self):
	    self.keyboard = Keys()
	    self.model = None

    def acceleration(self):
        self.keyboard.PressKey(N)
        time.sleep(0.01)
        self.keyboard.ReleaseKey(A)
        self.keyboard.ReleaseKey(D)
        #time.sleep(0.2)

    def left_acceleration(self):
        self.keyboard.PressKey(N)
        time.sleep(0.01)
        self.keyboard.PressKey(A)
        self.keyboard.ReleaseKey(D)
        #time.sleep(0.2)

    def right_acceleration(self):
        self.keyboard.PressKey(N)
        self.keyboard.PressKey(D)
        self.keyboard.ReleaseKey(A)
        #time.sleep(0.2)
		
    def jump(self):
        self.keyboard.PressKey(SPACE)
        time.sleep(0.01)
        self.keyboard.ReleaseKey(SPACE)
	
    def load_model(self, model):
        self.model = model
	
    def predict_action(self, image):
        image = image.astype('float32') / 255
        return self.model.predict(image)
		
if __name__ == '__main__':
    agent = Agent('test')
    agent.left()
    agent.right()
    agent.up()
    agent.down()
    agent.shoot()
    agent.pass_defend()
    agent.release_actions()