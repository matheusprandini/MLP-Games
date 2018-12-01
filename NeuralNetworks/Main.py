import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from CollectDataDQN.CatchGame import CatchGame
from CollectDataDQN.SnakeGame import SnakeGame
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
	
def preprocess_images_RGB(image, size):
	
	# single image
	x_t = image
	x_t = imresize(x_t, (size, size, 3))
	x_t = x_t.astype("float")
	x_t /= 255.0
	
	x_t = np.expand_dims(x_t, axis=0)

	return np.reshape(x_t, (1, size*size*3))
	
def preprocess_images_grayscale(image, size):
	
	# single image
	#x_t = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_t = image
    x_t = imresize(x_t, (size, size, 1))
    x_t = x_t.astype("float")
    x_t /= 255.0
	
    x_t = np.expand_dims(x_t, axis=0)

    return np.reshape(x_t, (1, size*size*1))

def play_game(game_env, model, number_games):
	
    num_games, num_wins = 0, 0

    for e in range(number_games):
        loss = 0.0
        size = 32
        game_env.reset()

	    # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, x_t1, r_0, game_over = game_env.step(a_0) 
        s_t = preprocess_images_RGB(x_t, size)
        #s_t = preprocess_images_grayscale(x_t1, size)

        while not game_over:
            s_tm1 = s_t
		    # next action
            q = model.predict(s_t)
            a_t = np.argmax(q)
		    # apply action, get reward
            x_t, x_t1, r_t, game_over = game_env.step(a_t)
            s_t = preprocess_images_RGB(x_t, size)
           #s_t = preprocess_images_grayscale(x_t1, size)
			
		    # if reward, increment num_wins
            if r_t == 1:
                num_wins += 1

        num_games += 1
        print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="\r")
		
def play_snake_game(game_env, model, number_games):
	
    num_games, total_score, num_wins = 0, 0, 0

    for e in range(number_games):
        loss = 0.0
        size = 32
        game_env.reset()

	    # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over, _ = game_env.step(a_0) 
        s_t = preprocess_images_RGB(x_t, size)
        #s_t = preprocess_images_grayscale(x_t, size)
        cv2.imwrite('snake.png',x_t)
        while not game_over:
            s_tm1 = s_t
			
            # Random (exploration)
            if np.random.rand() <= 0.05:
                a_t = np.random.randint(low=0, high=4, size=1)[0]
            # Best action (exploitation)
            else:
                q = model.predict(s_t)
                a_t = np.argmax(q)

            print(q)
		    # apply action, get reward
            x_t, r_t, game_over, score = game_env.step(a_t)
            s_t = preprocess_images_RGB(x_t, size)
            #s_t = preprocess_images_grayscale(x_t, size)

            if r_t == 1:
                num_wins += 1
			
        num_games += 1
        total_score += score
        print("Game: {:03d}, Win Count: {:03d}".format(num_games, num_wins))
        #print("Game: {:03d}, Score: {:03d} | Avg Score: {:.5f}".format(num_games, score, float(total_score / (e+1))))
		

# Accuracy = 85% Victory = 95%
def mlp_train_catch_game():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(512)
    mlp.create_layer(512)
    mlp.create_layer(3)
	
    #Load data
    data = Data("training_data_catch_game_dqn_1.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data(3, 1)

    epochs, mse = mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 1000, 0.1, 1e-10, True)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))
	
    plt.plot(epochs, mse)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.savefig('mse_catch.png')
	
    np.save("model_test_catch_1.npy", mlp.parameters)


def mlp_plays_catch_game():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(3)

    parameters = np.load(os.path.join("TrainedModels", "model_test_catch_grayscale.npy"))
    mlp.load_parameters(parameters.item())
	
    catch_game = CatchGame()

    play_game(catch_game, mlp, 100)
	
# Accuracy = % Victory = %
def mlp_train_catch_game_RGB():
    mlp = MLP()

    mlp.create_layer(3072)
    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(3)
	
    #Load data
    data = Data("training_data_catch_game_dqn_rgb.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data(3, 3)

    epochs, mse = mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 1000, 0.1, 1e-10, True)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))
	
    plt.plot(epochs, mse)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.savefig('mse_catch_rgb_1.png')
	
    np.save("model_test_catch_rgb.npy", mlp.parameters)
	
def mlp_plays_catch_game_RGB():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(3)

    parameters = np.load(os.path.join("TrainedModels", "model_test_catch_rgb.npy"))
    mlp.load_parameters(parameters.item())
	
    catch_game = CatchGame()

    play_game(catch_game, mlp, 100)


## Accuracy: 81%
def mlp_train_fifa_game_RGB():
    mlp = MLP()

    mlp.create_layer(3072)
    mlp.create_layer(512)
    mlp.create_layer(512)
    mlp.create_layer(512)
    mlp.create_layer(512)
    mlp.create_layer(4)
	
    #Load data
    data = Data("data_fifa_4actions_RGB.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data(4, 3)

    epochs, mse = mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 1000, 0.1, 1e-10, True)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))
	
    plt.plot(epochs, mse)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.savefig('mse_fifa.png')
	
    np.save("model_test_fifa.npy", mlp.parameters)

## Accuracy: 99.5% Avg Score: 
def mlp_train_snake_game_RGB():
    mlp = MLP()

    mlp.create_layer(3072)
    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(4)
	
    #Load data
    data = Data("training_data_snake_game_rgb.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data(4, 3)

    epochs, mse = mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 1000, 0.4, 1e-10, True)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))
	
    plt.plot(epochs, mse)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.savefig('mse_snake.png')
	
    np.save("model_test_snake.npy", mlp.parameters)
	
def mlp_plays_snake_game_RGB():
    mlp = MLP()

    mlp.create_layer(3072)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(4)

    parameters = np.load(os.path.join("TrainedModels", "model_test_snake_rgb.npy"))
    mlp.load_parameters(parameters.item())
	
    snake_game = SnakeGame()

    play_snake_game(snake_game, mlp, 100)

## Accuracy: 81.91%
def mlp_train_snake_game_grayscale():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(1024)
    mlp.create_layer(4)
	
    #Load data
    data = Data("training_data_snake_game_grayscale.npy")

    # Preprocess data
    inputs, labels = data.preprocessing_data(4, 1)

    epochs, mse = mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 1000, 0.4, 1e-10, True)
	
    print('Accuracy: ', mlp.compute_accuracy(inputs, labels))
	
    plt.plot(epochs, mse)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio')
    plt.savefig('mse_snake_grayscale.png')
	
    np.save("model_test_snake_grayscale.npy", mlp.parameters)
	
def mlp_plays_snake_game_grayscale():
    mlp = MLP()

    mlp.create_layer(1024)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(1)
    mlp.create_layer(4)

    parameters = np.load(os.path.join("TrainedModels", "model_test_snake_grayscale.npy"))
    mlp.load_parameters(parameters.item())
	
    snake_game = SnakeGame()

    play_snake_game(snake_game, mlp, 100)

####################################### MAIN #######################################

#mlp_train_catch_game()
#mlp_plays_catch_game()

#mlp_train_catch_game_RGB()
#mlp_plays_catch_game_RGB()

#mlp_train_snake_game_RGB()
mlp_plays_snake_game_RGB()

#mlp_train_snake_game_grayscale()
#mlp_plays_snake_game_grayscale()