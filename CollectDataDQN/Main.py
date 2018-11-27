from DQN import DQN
from DQN_RGB import DQN_RGB
from CatchGame import CatchGame

# Train or Test DQN agents

#dqn = DQN()
dqn = DQN_RGB()
catch_game = CatchGame()

#dqn.train_model(catch_game)
dqn.test_model(catch_game, "rl-network-snake2-RGB")
