from __future__ import division, print_function
from keras.models import *
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy.misc import imresize
import collections
import numpy as np
import os

# Initialize Global Parameters
DATA_DIR = ""
NUM_ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
MEMORY_SIZE = 50000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 500
NUM_EPOCHS_TRAIN = 10000
NUM_EPOCHS_TEST = 10

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

class DQN:

    def __init__(self):
        self.model = self.build_model()
        self.experience = collections.deque(maxlen=MEMORY_SIZE)
		
    def load_model(self, file_name):
        self.model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")))

    # build the model
    def build_model(self):
	
		# Sequential Model
        model = Sequential()
		
		# 1st cnn layer
        model.add(Conv2D(84, kernel_size=8, strides=4, 
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(84, 84, 1)))
        model.add(Activation("relu"))
		
        # 2st cnn layer
        model.add(Conv2D(64, kernel_size=4, strides=2, 
                 kernel_initializer="normal", 
                 padding="same"))
        model.add(Activation("relu"))
		
		# 3st cnn layer
        model.add(Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer="normal",
                 padding="same"))
        model.add(Activation("relu"))
		
		# Flattening parameters
        model.add(Flatten())
		
		# 1st mlp layer
        model.add(Dense(512, kernel_initializer="normal"))
        model.add(Activation("relu"))
		
		# 2st (last) cnn layer -> Classification layer
        model.add(Dense(NUM_ACTIONS, kernel_initializer="normal"))

		
		# Compiling Model
        model.compile(optimizer=Adam(lr=1e-6), loss="mse")

		# Show model details
        model.summary()
		
        return model

    # Preprocess images and stacks in a deque
    def preprocess_images(self,image):
        
		# single image
        x_t = image
        x_t = imresize(x_t, (84, 84, 1))
        x_t = x_t.astype("float")
        x_t /= 255.0
		
		# convert image
        x_image = imresize(image, (32, 32, 1))
        x_image = x_image.astype("float")
        x_image /= 255.0
		
        x_t = np.expand_dims(x_t, axis=0)

        return np.reshape(x_t, (1, 84, 84, 1)), np.reshape(np.expand_dims(x_image, axis=0), (1, 32, 32, 1))
	
	# Return a batch of experiencie to train the dqn model
    def get_next_batch(self, num_actions, gamma, batch_size):
        batch_indices = np.random.randint(low=0, high=len(self.experience),
                                      size=batch_size)
        batch = [self.experience[i] for i in batch_indices]
        X = np.zeros((batch_size, 84, 84, 1))
        Y = np.zeros((batch_size, num_actions))
		
        # Building the batch data
        for i in range(len(batch)):
            s_t, a_t, r_t, s_tp1, game_over = batch[i]
            X[i] = s_t
            Y[i] = self.model.predict(s_t)[0]
            Q_sa = np.max(self.model.predict(s_tp1)[0])
            if game_over:
                Y[i, a_t] = r_t
            else:
                Y[i, a_t] = r_t + gamma * Q_sa

        return X, Y
		
	# Train the dqn
    def train_model(self, game_env):

		# Initializing experience memory
        self.experience = collections.deque(maxlen=MEMORY_SIZE)
        
        num_games, num_wins = 0, 0
        epsilon = INITIAL_EPSILON

        for e in range(NUM_EPOCHS):
            loss = 0.0
            game_env.reset()
    
            # get first state
            a_0 = 1  # (0 = up, 1 = right, 2 = down, 3 = left)
            x_t, r_0, game_over = game_env.step(a_0) 
            s_t = self.preprocess_images(x_t)
	
            while not game_over:
                s_tm1 = s_t
                # Get next action
				
				# Observation action (random)
                if e <= NUM_EPOCHS_OBSERVE:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                # Random or the best current action based on q-value (dqn model)
                else:
					# Random (exploration)
                    if np.random.rand() <= epsilon:
                        a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                    # Best action (exploitation)
                    else:
                        q = self.model.predict(s_t)[0]
                        a_t = np.argmax(q)
                
                # apply action, get reward
                x_t, r_t, game_over = game_env.step(a_t)
                s_t = self.preprocess_images(x_t)
        
		        # if reward, increment num_wins
                if r_t == 1:
                    num_wins += 1
                # store experience
                self.experience.append((s_tm1, a_t, r_t, s_t, game_over))
        
                if e > NUM_EPOCHS_OBSERVE:
                    # finished observing, now start training
                    # get next batch
                    X, Y = self.get_next_batch(NUM_ACTIONS, GAMMA, BATCH_SIZE)
                    loss += self.model.train_on_batch(X, Y)
        
            # reduce epsilon gradually
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS
        
            print("Epoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}"
                .format(e + 1, NUM_EPOCHS, loss, num_wins))
				
            if e % 100 == 0:
                self.model.save(os.path.join(DATA_DIR, "rl-network-screenshot.h5"), overwrite=True)
        
        self.model.save(os.path.join(DATA_DIR, "rl-network-screenshot.h5"), overwrite=True)
		
	# Test dqn model
    def test_model(self, game, file_name):
        model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")))
        
        num_games, num_wins = 0, 0

        for e in range(NUM_EPOCHS_TEST):
            loss = 0.0
            game.reset()
    
            # get first state
            a_0 = 1  # (0 = left, 1 = stay, 2 = right)
            x_t, r_0, game_over = game.step(a_0) 
            s_t, _ = self.preprocess_images(x_t)

            while not game_over:
                s_tm1 = s_t
                # next action
                q = model.predict(s_t)[0]
                a_t = np.argmax(q)
                # apply action, get reward
                x_t, r_t, game_over = game.step(a_t)
                s_t, _ = self.preprocess_images(x_t)
                # if reward, increment num_wins
                if r_t == 1:
                    num_wins += 1

            num_games += 1
            print("Game: {:03d}, Wins: {:03d}".format(num_games, num_wins), end="\r")
        
print("")
		
if __name__ == '__main__':
	
    dqn = DQN()