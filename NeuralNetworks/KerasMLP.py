from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import *
from keras import layers
from keras import models
from scipy.misc import imresize
import numpy as np
import keras
import sys
import cv2
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from CollectDataDQN.CatchGame import CatchGame
from CollectDataDQN.SnakeGame import SnakeGame
from CollectDataDQN.Data import Data

# Preprocess images and stacks in a deque
def preprocess_images(image, size):
	
	# single image
    x_t = image
    #x_t = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_t = imresize(x_t, (size, size, 3))
    x_t = x_t.astype("float")
    x_t /= 255.0
	
    x_t = np.expand_dims(x_t, axis=0)

    return np.reshape(x_t, (1, size*size*3))
	
class KerasMLP():

    def __init__(self, input_neurons=1024, output_neurons=4):
        self.model = self.build_model(input_neurons, output_neurons)
        self.num_actions = output_neurons
		
    def build_model(self, input_neurons, output_neurons):
        # Building a mlp
        model = models.Sequential()
        model.add(layers.Dense(1024, activation='relu', input_shape=(input_neurons,)))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(output_neurons, activation='sigmoid'))

		# Show model's parameters
        model.summary()
		
        return model
		
    def train_model(self, data_file, model_name):
        #Load data
        data = Data(data_file)

		# Preprocess data
        inputs, labels = data.preprocessing_data(self.num_actions, 3)

        print(labels)

        ## Random sampling (80:20)

        number_examples = inputs.shape[0]

        split = int(0.85 * number_examples)

        x_train = inputs[:split]
        x_test = inputs[split:]
        y_train = labels[:split]
        y_test = labels[split:]

        # Training the mlp
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
        self.model.fit(inputs, labels, epochs=50, batch_size=64)
		
        self.test_model(inputs, labels)
		
        # Save model in "TrainedModels" folder
        self.model.save(os.path.join("TrainedModels", model_name), overwrite=True)
		
    def train_model2(self, inputs, labels, model_name):

        ## Random sampling (80:20)

        number_examples = inputs.shape[0]

        split = int(0.8 * number_examples)

        inputs = inputs.astype('float32') / 255
		
        x_train = inputs[:split]
        x_test = inputs[split:]
        y_train = labels[:split]
        y_test = labels[split:]
		
        # Training the mlp
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=50, batch_size=64)
		
        self.test_model(x_test, y_test)
		
        # Save model in "TrainedModels" folder
        self.model.save(os.path.join("TrainedModels", model_name), overwrite=True)
		
    def test_model(self, test_inputs, test_labels):
        # Testing the mlp
        test_loss, test_acc = self.model.evaluate(test_inputs, test_labels)

        print('Keras MLP loss:', test_loss)
        print('Keras MLP accuracy:', test_acc)
		
    def predict(self, input):
        return self.model.predict(input)

if __name__ == "__main__":	
    kerasMLP = KerasMLP(1024*3)
    kerasMLP.train_model("data_fifa_4actions_RGB.npy", "keras-mlp-fifa-32x32_rgb")

    '''#catch_game = CatchGame()
    snake_game = SnakeGame()
	
    num_games, num_wins = 0, 0

    #game = catch_game
    game = snake_game

    for e in range(10):
        loss = 0.0
        size = 32
        score = 0
        game.reset()

	    # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over, _ = game.step(a_0) 
        s_t = preprocess_images(x_t, size)

        while not game_over:
            s_tm1 = s_t
		    # next action
            q = kerasMLP.predict(s_t)[0]
            print(q)
            a_t = np.argmax(q)
		    # apply action, get reward
            x_t, r_t, game_over, score = game.step(a_t)
            s_t = preprocess_images(x_t, size)
		    # if reward, increment num_wins
            if r_t == 1:
                num_wins += 1

        num_games += 1
        #print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="\r")
        print("Game: {:03d}, Score: {:03d}".format(num_games, score))'''