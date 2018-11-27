import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))

from CollectDataDQN.SnakeGame import SnakeGame
from CollectDataDQN.DQN_RGB import DQN_RGB
from random import shuffle
from scipy.misc import imresize
import numpy as np
import keras
import time
import cv2

class Data():

    def __init__(self, file_name='data.npy'):
        self.file_name = file_name
        self.path_file = 'Data/' + file_name
        self.data = self.initialize_data()

    # Load data if exists, else return an empty list
    def initialize_data(self):
        if os.path.isfile(self.path_file):
            print('File exists, loading previous data!')
            return list(np.load(self.path_file))
        print('File does not exist, creating file!')
        return []
		
    def preprocess_image_data(self, image):
        x_t = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_t = imresize(image, (32, 32, 3))

    def collect_data(self, game, file_name, number_games):

        dqn = DQN()
        dqn.load_model(file_name)
	
        num_wins = 0

        for episode in range(number_games):
            game.reset()
    
            # get first state
            a_0 = 1 
            x_t, r_0, game_over = game.step(a_0)
            s_t, s_image = dqn.preprocess_images(x_t)

            while not game_over:

                # next action
                
                # Random (exploration)
                if np.random.rand() <= epsilon:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                # Best action (exploitation)
                else:
                    q = self.model.predict(s_t)[0]
                    a_t = np.argmax(q)
				
                self.data.append([s_image, a_t])
				
                # apply action, get reward
                x_t, r_t, game_over = game.step(a_t)
                s_t, s_image = dqn.preprocess_images(x_t)
                # if reward, increment num_wins
                if r_t == 1:
                    num_wins += 1
					
            if (episode + 1) % 100 == 0:
                np.save(self.path_file,self.data)
					
            print("Game: {:03d}, Wins: {:03d}".format(episode+1, num_wins), end="\r")
        
        np.save(self.path_file,self.data)
		
    def collect_data_DQN_RGB(self, game, file_name, num_actions, number_games):

        dqn = DQN_RGB()
        dqn.load_model(file_name)
	
        num_wins = 0

        for episode in range(number_games):
            game.reset()
            score = 0
    
            # get first state
            a_0 = 1 
            x_t, r_0, game_over, _ = game.step(a_0)
            s_t, s_image = dqn.preprocess_image(x_t)

            while not game_over:

                # next action
                
                # Random (exploration)
                if np.random.rand() <= 0.05:
                    a_t = np.random.randint(low=0, high=num_actions, size=1)[0]
                # Best action (exploitation)
                else:
                    q = dqn.model.predict(s_t)[0]
                    a_t = np.argmax(q)
				
                self.data.append([s_image, a_t])
				
                # apply action, get reward
                x_t, r_t, game_over, score = game.step(a_t)
                s_t, s_image = dqn.preprocess_image(x_t)
                
				# if reward, increment num_wins
                if r_t == 1:
                    num_wins += 1
					
            if (episode + 1) % 5 == 0:
                np.save(self.path_file,self.data)
					
            #print("Game: {:03d}, Wins: {:03d}".format(episode+1, num_wins), end="\r")
            print("Game: {:03d}, Score: {:03d}".format(episode+1, score))
        
        np.save(self.path_file,self.data)
			
    # Show all data in training data and separate in input and output data (image, output_move)
    def generate_input_and_output_data(self):
        input_data = []
        output_data = []
		
        shuffle(self.data)
		
        for data in self.data:
            img = data[0]
            output_move = data[1]
            input_data.append(img)
            output_data.append(output_move)
		
        return input_data, output_data
	
	# Convert numeric labels to categorical (one-hot-enconding)
    def to_categorical(self, labels, number_actions):
	
        categorical_labels = keras.utils.to_categorical(labels, number_actions)
		
        return categorical_labels
		
    # Preprocessing the data (Generate and normalizing)
    def preprocessing_data(self, number_actions):
	
	    ## Separate input and output data
		
        inputs, labels = self.generate_input_and_output_data()
	
        ## Reshaping and Normalizing data

        new_inputs = np.array([i for i in inputs]).reshape(-1,32*32*3)

        new_labels = [i for i in self.to_categorical(labels, number_actions)]
        new_labels = np.array(new_labels)

        return new_inputs, new_labels

		
if __name__ == "__main__":
    '''data_catch_game = Data("training_data_catch_game_dqn_1.npy")
    catch_game = CatchGame()
    t = time.process_time()
    data_catch_game.collect_data(catch_game, "rl-network-screenshot-catch-5000", 1)
    elapsed_time = time.process_time() - t
    print("Time elapsed: ", elapsed_time, "s")'''
    data_snake_game = Data("training_data_snake_game_rgb.npy")
    snake_game = SnakeGame()
    t = time.process_time()
    data_snake_game.collect_data_DQN_RGB(snake_game, "rl-network-snake2-RGB", 4, 100)
    elapsed_time = time.process_time() - t
    print("Time elapsed: ", elapsed_time, "s")