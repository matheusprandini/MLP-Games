# MLP-Games
Using MLP to play simple games - Machine Learning Course - UFU

## Getting Started

Instructions to run the project...

### Prerequisites (Packages)
```
Numpy: pip install numpy
Scipy: pip install scipy
Keras: pip install Keras (Needs Tensorflow: https://www.tensorflow.org/install/)
```

### Installing

Just clone the repo :)

```
git clone https://github.com/matheusprandini/MLP-Games.git
```

### Usage

#### Module 1: **CollectDataDQN** 

Module responsible to collect data from the catch game using an agent implemented with dqn technique.

- **Data folder**: collected data files (move to the Data folder in the **NeuralNetworks** module to use them)
- **CatchGame.py**: game environment.
- **DQN.py**: agent implemented with dqn technique.
- **Data.py**: responsible for collecting, saving and preprocessing the data from the interaction between the dqn agent and the catch game.
- **Main.py**: train or test a dqn agent.

- **rl-network-screenshot-catch-5000.h5**: dqn agent's model (trained with 5000 episodes).

Commands:

- Collect Data (just need to pass the number of games to collect data in the code):
```
python Data.py
```
**Obs**: moves the collected data to the Data folder in the NeuralNetworks module.

- Train or test a dqn agent (comment or uncomment the desired row in the code):
```
python Main.py
```

#### Module 2: **NeuralNetworks** 

Module containing implemented neural networks.

- **Data folder**: contains the data used to train a model.
- **TrainedModels folder**: contains the trained models.
- **DynamicMLP.py**: mlp (with arbitrary number of hidden layers) implemented using only numpy.
- **KerasMLP.py**: mlp implemented with the Keras (built on top of the tensorflow) framework (used to validate data).
- **Main.py**: train or test a mlp model.

Commands:

- Training a Keras Model to play the catch game:
```
python KerasMLP.py
```

- Training or Testing the Dynamic MLP on the catch game (comment or uncomment the desired row in the code):
```
python Main.py
```

## Authors

* **Alexandre Henrique** - [AlexandreH13](https://github.com/AlexandreH13)
* **Guilherme Silva Alves** - [guilhermebreed](https://github.com/guilhermebreed)
* **Matheus Prandini** - [matheus_prandini](https://github.com/matheusprandini)
* **Mônica Ribeiro** - [monicarib](https://github.com/monicarib)
