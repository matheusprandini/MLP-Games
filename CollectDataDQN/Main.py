from DQN import DQN
from CatchGame import CatchGame

# Train or Test DQN agents

dqn = DQN()
catch_game = CatchGame()

#dqn.train_model(catch_game)
dqn.test_model(catch_game, "rl-network-screenshot-catch-5000")
