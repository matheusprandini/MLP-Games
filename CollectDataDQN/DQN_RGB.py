from __future__ import division, print_function
from keras.models import *
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import *
from scipy.misc import imresize
import collections
import numpy as np
import cv2
import os
import tensorflow as tf

# Initialize Global Parameters
DATA_DIR = "Models/"
NUM_ACTIONS = 4 # number of valid actions (up, down, right,left)
GAMMA = 0.9 # decay rate of past observations
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
MEMORY_SIZE = 50000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 1000
NUM_EPOCHS_TRAIN = 10000
NUM_EPOCHS_TEST = 100

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

class DQN_RGB:

    def __init__(self):
        self.model = self.build_model()
        self.experience = collections.deque(maxlen=MEMORY_SIZE)
		
    # Load Model (Main DQN)
    def load_model(self, file_name, huber_loss=True):
        if huber_loss:
            self.model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")), custom_objects={'huber_loss': self.huber_loss})
        else:
            self.model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")))
		
    # Save Model (Main DQN)
    def save_model(self, file_name):
        self.model.save(os.path.join(DATA_DIR, (file_name + ".h5")), overwrite=True)
		
    # Save DQN model and update Target Model
    def update_target_network(self):
        self.save_model("target_network")
        self.target_model = load_model(os.path.join(DATA_DIR, ("target_network.h5")), custom_objects={'huber_loss': self.huber_loss})
		
    # Defining the huber loss
    def huber_loss(self, y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)

    # build the model
    def build_model(self):
	
		# Sequential Model
        model = Sequential()
		
		# 1st cnn layer
        model.add(Conv2D(32, kernel_size=8, strides=4, 
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(84, 84, 3)))
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
		
		# 2st (last) cnn layer -> Classification layer (up, down, right, left)
        model.add(Dense(NUM_ACTIONS, kernel_initializer="normal"))

		
		# Compiling Model
        model.compile(optimizer=RMSprop(lr=1e-4), loss=self.huber_loss)

		# Show model details
        model.summary()
		
        return model

    # Preprocess image
    def preprocess_image(self,image):

        #x_t = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_t = imresize(image, (84, 84, 3))
        x_t = x_t.astype("float")
        x_t /= 255.0
        
        s_t = np.expand_dims(x_t, axis=0)
		
        # convert image
        x_image = image
        #x_image = cv2.cvtColor(x_image, cv2.COLOR_BGR2GRAY)
        #x_image = imresize(x_image, (32, 32, 1))
        x_image = imresize(x_image, (32, 32, 3))
        x_image = x_image.astype("float")
        x_image /= 255.0
        #x_image = np.reshape(np.expand_dims(x_image, axis=0), (1, 32, 32, 1))
        x_image = np.reshape(np.expand_dims(x_image, axis=0), (1, 32, 32, 3))

        return s_t, x_image
	
	# Return a batch of experiencie to train the dqn model
    def get_next_batch(self, num_actions, gamma, batch_size):
        batch_indices = np.random.randint(low=0, high=len(self.experience),
                                      size=batch_size)
        batch = [self.experience[i] for i in batch_indices]
        X = np.zeros((batch_size, 84, 84, 3))
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
            score = 0
            game_env.reset()
    
            # get first state
            a_0 = 1  # (0 = up, 1 = right, 2 = down, 3 = left)
            x_t, r_0, game_over, _ = game_env.step(a_0) 
            s_t = self.preprocess_image(x_t)
	
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
                x_t, r_t, game_over, score = game_env.step(a_t)
                s_t = self.preprocess_image(x_t)
        
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
            if e > NUM_EPOCHS_OBSERVE and epsilon > FINAL_EPSILON:
                epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON) / (NUM_EPOCHS_TRAIN / 1.5))
        
            print("Epoch {:04d}/{:d} | Epsilon: {:.5f} | Loss {:.5f} | Score: {:d}"
                .format(e + 1, NUM_EPOCHS, epsilon, loss, score))
				
            if e % 100 == 0:
                self.save_model("rl-network-screenshot")
        
        self.save_model("rl-network-screenshot")
		
	# Test dqn model
    def test_model(self, game, file_name):
        self.load_model(file_name)
        
        num_games = 0

        for e in range(NUM_EPOCHS_TEST):
            loss = 0.0
            score = 0
            game.reset()
    
            # get first state
            a_0 = 1
            x_t, r_0, game_over, _ = game.step(a_0) 
            s_t = self.preprocess_image(x_t)

            while not game_over:
                s_tm1 = s_t
				
                # Random (exploration)
                if np.random.rand() <= epsilon:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                # Best action (exploitation)
                else:
                    q = self.model.predict(s_t)[0]

                a_t = np.argmax(q)
                # apply action, get reward
                x_t, r_t, game_over, score = game.step(a_t)
                s_t = self.preprocess_image(x_t)

            num_games += 1
            print("Game: {:03d}, Score: {:03d}".format(num_games, score))
        
print("")
		
if __name__ == '__main__':
	
    dqn = DQN_RGB()