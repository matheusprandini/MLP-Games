import sys
import os

sys.path.insert(0, os.path.join(sys.path[0]))

from CollectDataDQN.CatchGame import CatchGame
from CollectDataDQN.DQN import DQN
from random import shuffle
import numpy as np
import keras
import time

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
                q = dqn.model.predict(s_t)[0]
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
    def to_categorical(self, labels):
	
        categorical_labels = keras.utils.to_categorical(labels, 3)
		
        return categorical_labels
		
    # Preprocessing the data (Generate and normalizing)
    def preprocessing_data(self):
	
	    ## Separate input and output data
		
        inputs, labels = self.generate_input_and_output_data()
	
        ## Reshaping and Normalizing data

        new_inputs = np.array([i for i in inputs]).reshape(-1,32*32)

        new_labels = [i for i in self.to_categorical(labels)]
        new_labels = np.array(new_labels)

        return new_inputs, new_labels

		
if __name__ == "__main__":
    data_catch_game = Data("training_data_catch_game_dqn_1.npy")
    catch_game = CatchGame()
    t = time.process_time()
    data_catch_game.collect_data(catch_game, "rl-network-screenshot-catch-5000", 0)
    elapsed_time = time.process_time() - t
    print("Time elapsed: ", elapsed_time, "s")