import sys
import os
import numpy as np
from scipy.misc import imresize

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from CollectDataDQN.CatchGame import CatchGame
from CollectDataDQN.Data import Data
from DynamicMLP import MLP

# Preprocess images and stacks in a deque
def preprocess_images(image, size):
	
	# single image
	x_t = image
	x_t = imresize(x_t, (size, size, 1))
	x_t = x_t.astype("float")
	x_t /= 255.0
	
	x_t = np.expand_dims(x_t, axis=0)

	return np.reshape(x_t, (1, size*size))

def play_game(game_env, model, number_games):
	
    num_games, num_wins = 0, 0

    for e in range(number_games):
	    loss = 0.0
	    size = 32
	    game_env.reset()

	    # get first state
	    a_0 = 1  # (0 = left, 1 = stay, 2 = right)
	    x_t, r_0, game_over = game_env.step(a_0) 
	    s_t = preprocess_images(x_t, size)

	    while not game_over:
		    s_tm1 = s_t
		    # next action
		    q = model.predict(s_t)
		    print(q)
		    a_t = np.argmax(q)
		    # apply action, get reward
		    x_t, r_t, game_over = game_env.step(a_t)
		    s_t = preprocess_images(x_t, size)
		    # if reward, increment num_wins
		    if r_t == 1:
			    num_wins += 1

	    num_games += 1
	    print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="\r")
		

def mlp_train_catch_game():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(512)
    mlp.create_layer(3)
	
    #Load data
    data = Data("training_data_catch_game_dqn_1.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data()

    mlp.training_mlp_any_layers_with_number_iterations(inputs[:10000], labels[:10000], 10, 0.1)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))


def mlp_plays_catch_game():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(512)
    mlp.create_layer(3)

    parameters = np.load(os.path.join("TrainedModels", "model_mlp-4-20000-01.npy"))
    mlp.load_parameters(parameters.item())
	
    catch_game = CatchGame()

    play_game(catch_game, mlp, 100)

####################################### MAIN #######################################

mlp_train_catch_game()
#mlp_plays_catch_game()