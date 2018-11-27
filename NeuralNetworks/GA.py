from DynamicMLP import MLP
from random import shuffle
from random import randint
import numpy as np
import math
import time

class IndividualNN():
    
    def __init__(self, fitness=0):
        self.neural_network = None
        self.number_hidden_layers = 0 # Number of hidden layers 
        self.neurons_hidden_layers = 0 # Number of neurons per hidden layer
        self.learning_rate = 0 # Learning rate
        self.fitness = 0 # Accuracy on the test set
        self.training_time = 0 # Time spent to train the NN

    def __lt__(self, other_individual):
        if self.fitness != other_individual.fitness:
            return self.fitness > other_individual.fitness
        elif self.training_time != other_individual.training_time:
            return self.training_time < other_individual.training_time
        else:
            return str(self.neural_network) < str(other_individual.neural_network)

    def initialize_individual(self, dataset_inputs, dataset_labels):
        
        #dataset = dataset_file[:]

        neurons_layer = [64,128,256,512, 1024]
        learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.4]
        
        if self.number_hidden_layers == 0:
            self.number_hidden_layers = randint(1,4)
			
        if self.neurons_hidden_layers == 0:
            self.neurons_hidden_layers = neurons_layer[randint(0,len(neurons_layer)-1)]
			
        if self.learning_rate == 0:
            self.learning_rate = learning_rates[randint(0,len(learning_rates)-1)]
        
        self.neural_network = self.initialize_neural_network(dataset_inputs.shape[1], dataset_labels.shape[1])

        self.fitness, self.training_time = self.compute_fitness(dataset_inputs, dataset_labels)
        
    def initialize_neural_network(self, input_neurons, output_neurons):
         
        ## Create NN
        mlp = MLP()
        
        # Input Layer
        mlp.create_layer(input_neurons)
        
        # Hidden Layers
        for i in range(self.number_hidden_layers):
            mlp.create_layer(self.neurons_hidden_layers)
            
        # Output Layer
        mlp.create_layer(output_neurons)
        
        return mlp
        
    def compute_fitness(self, dataset_inputs, dataset_labels):
        
        last_time = time.time()
        
        # Training the MLP
        self.neural_network.training_mlp_any_layers_with_number_iterations(dataset_inputs[:], dataset_labels[:], 200, self.learning_rate, log_mse=False)
        
        total_training_time = (time.time()-last_time)
        
        test_accuracy = self.neural_network.compute_accuracy(dataset_inputs[:], dataset_labels[:])
        
        return test_accuracy, total_training_time
		

import heapq
import random

class PopulationNN():
    
    def __init__(self, size):
        self.size = size # Population size
        self.individuals = [] # Population's individuals
        
    def initialize_population(self, dataset_inputs, dataset_labels):
        
        for i in range(self.size):
            ind = IndividualNN()
            ind.initialize_individual(dataset_inputs, dataset_labels)
            heapq.heappush(self.individuals, ind)
    
    # Returns the best individual
    def get_best_individual(self):
        # Best individual -> Position 0 in the heap
        return self.individuals[0] 
    
    def selects_individuals_crossover(self, crossover_rate):
        
        selected_individuals = []
        
        number_individuals = (int) (self.size * (crossover_rate / 100.0))
        
        # Even number
        if (number_individuals % 2 != 0):
            number_individuals -= 1
        
        # Selects individuals through tournaments
        for i in range(number_individuals):
            
            ind = self.tournament_selection()

            selected_individuals.append(ind)

        return selected_individuals
    
    # Tournament between 3 individuals
    def tournament_selection(self):
        
        tournament_selected_individuals = []
        tournament_best_individual = None
        
        for i in range(3):
            tournament_selected_individuals.append(random.choice(self.individuals))
        
        for ind in tournament_selected_individuals:
            if (tournament_best_individual == None) or (ind.fitness >= tournament_best_individual.fitness and ind.training_time < tournament_best_individual.training_time):
                tournament_best_individual = ind

        return tournament_best_individual
		
    def execute_crossover(self, individual1, individual2, dataset_inputs, dataset_labels):
	
        children1 = IndividualNN()
        children2 = IndividualNN()
        
        # Crossover mask
        mask = randint(0,2)
        
        if mask == 0:
            children1.number_hidden_layers = individual1.number_hidden_layers
            children1.neurons_hidden_layers = individual2.neurons_hidden_layers
            children1.learning_rate = individual2.learning_rate
            children2.number_hidden_layers = individual2.number_hidden_layers
            children2.neurons_hidden_layers = individual1.neurons_hidden_layers
            children2.learning_rate = individual1.learning_rate
            
        elif mask == 1:
            children1.number_hidden_layers = individual2.number_hidden_layers
            children1.neurons_hidden_layers = individual1.neurons_hidden_layers
            children1.learning_rate = individual2.learning_rate
            children2.number_hidden_layers = individual1.number_hidden_layers
            children2.neurons_hidden_layers = individual2.neurons_hidden_layers
            children2.learning_rate = individual1.learning_rate

        else:
            children1.number_hidden_layers = individual2.number_hidden_layers
            children1.neurons_hidden_layers = individual2.neurons_hidden_layers
            children1.learning_rate = individual1.learning_rate
            children2.number_hidden_layers = individual1.number_hidden_layers
            children2.neurons_hidden_layers = individual1.neurons_hidden_layers
            children2.learning_rate = individual2.learning_rate

        return children1, children2
		
    def execute_mutation(self, children, mutation_rate, dataset_inputs, dataset_labels):
    
        var1 = randint(1,10)
        var2 = randint(1,10)
        var3 = randint(1,10)
		
        if var1 <= (mutation_rate / 10):
            children.number_hidden_layers = 0
        if var2 <= (mutation_rate / 10):
            children.neurons_hidden_layers = 0
        if var3 <= (mutation_rate / 10):
            children.learning_rate = 0
		
        children.initialize_individual(dataset_inputs, dataset_labels)
                
        return children
		

import matplotlib.pyplot as plt
import random
from random import randint
import heapq

class GeneticAlgorithm():
    
    def __init__(self, number_generations, population_size, crossover_rate, mutation_rate, elitism_rate):
        self.number_generations = number_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
    '''
    GA NN
    '''

    def execute_ga_nn(self, dataset_inputs, dataset_labels, name):
    
        # Inicializa a População
        pop = PopulationNN(self.population_size)
        pop.initialize_population(dataset_inputs, dataset_labels)
        
        # Armazena o melhor indivíduo
        best_individual = pop.get_best_individual()
        best_generation = 1
        
        fitness_axis = []
        training_time_axis = []
        generations_axis = []

        # Executa o AG em numGeracoes
        for generation in range(1, self.number_generations+1):
            
            # Recupera o melhor indivíduo da geração
            best_individual_generation = pop.get_best_individual()
            
            fitness_axis.append(best_individual_generation.fitness)
            training_time_axis.append(best_individual_generation.training_time)
            generations_axis.append(generation)
            
            # Verifica se o melhor indivíduo recuperado é o melhor do AG até então
            if best_individual_generation.fitness > best_individual.fitness:
                best_individual = best_individual_generation
                best_generation = generation
            elif best_individual_generation.fitness == best_individual.fitness and best_individual_generation.training_time < best_individual.training_time:
                best_individual = best_individual_generation
                best_generation = generation
				
            print('Generation: ', generation)
            print('Best Individual: ', best_individual.fitness, ' - ', best_individual.training_time)
			
            if generation == self.number_generations:
                break
            
            #Método Torneio para selecionar Indivíduos para Crossover
            selected_individuals_crossover = pop.selects_individuals_crossover(self.crossover_rate)
            
            generated_children_crossover = []
    
            #Execução do Crossover Uniforme para geração de filhos
            for i in range(0,len(selected_individuals_crossover),2):
                children1, children2 = pop.execute_crossover(selected_individuals_crossover[i],selected_individuals_crossover[i+1], dataset_inputs, dataset_labels)

                generated_children_crossover.append(children1)
                generated_children_crossover.append(children2)

            #Execução de Mutação
            for children in generated_children_crossover:
                children = pop.execute_mutation(children, self.mutation_rate, dataset_inputs, dataset_labels)
                heapq.heappush(pop.individuals, children)
            
            new_population = []
            
            #Elistismo
            number_individuals_elitism = (int)((self.elitism_rate / 100) * self.population_size)
            
            for i in range(number_individuals_elitism):
                heapq.heappush(new_population, heapq.heappop(pop.individuals))
            
            for i in range(0,(self.population_size - number_individuals_elitism)):
                individual_position = randint(0,len(pop.individuals)-1)
                heapq.heappush(new_population, pop.individuals[individual_position])
                pop.individuals[individual_position] = pop.individuals[-1]
                pop.individuals.pop()  
            
            pop.individuals = new_population
        
        print('---------------')
        print("Best Individual: \n")
        print(best_individual.fitness, ' - ', best_individual.training_time)
        print(best_individual.neurons_hidden_layers)
        print(best_individual.number_hidden_layers)
        print(best_individual.learning_rate)
        print("Generation: ", best_generation)
        print('---------------')
        
        plt.plot(generations_axis, fitness_axis)
        plt.xlabel('Generations')
        plt.ylabel('Best Individual Fitness')
        plt.savefig(name + '_fitness_RGB.png')
		
        plt.plot(generations_axis, training_time_axis)
        plt.xlabel('Generations')
        plt.ylabel('Best Individual Training Time')
        plt.savefig(name + '_training_time_RGB.png')
		
        return best_individual
		
import keras
import os

# Show all data in training data and separate in input and output data (image, output_move)
def generate_input_and_output_data(data):
	input_data = []
	output_data = []
	
	shuffle(data)
	
	for d in data:
		img = d[0]
		output_move = d[1]
		input_data.append(img)
		output_data.append(output_move)
	
	return input_data, output_data

# Convert numeric labels to categorical (one-hot-enconding)
def to_categorical(labels, num_actions):

	categorical_labels = keras.utils.to_categorical(labels, num_actions)
	
	return categorical_labels
	
# Preprocessing the data (Generate and normalizing)
def preprocessing_data(data, num_actions, colors):

	## Separate input and output data
	
    inputs, labels = generate_input_and_output_data(data)

	## Reshaping and Normalizing data

    new_inputs = np.array([i for i in inputs]).reshape(-1,32*32*colors)

    new_labels = [i for i in to_categorical(labels, num_actions)]
    new_labels = np.array(new_labels)

    return new_inputs, new_labels
		
inputs_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs_xor = np.array([[1,0],[0,1],[0,1],[1,0]])

start = time.time()
ga = GeneticAlgorithm(10, 10, 80, 10, 20)
best_ind = ga.execute_ga_nn(inputs_xor, outputs_xor, 'xor')
np.save("model_test_xor.npy", best_ind.neural_network.parameters)
print('Time elapsed: ', (time.time() - start))

'''data = list(np.load('training_data_catch_game_dqn_1.npy'))
inputs, labels = preprocessing_data(data, 3, 1)

print(inputs.shape)
print(labels.shape)

start = time.time()
ga = GeneticAlgorithm(10, 10, 80, 10, 20)
best_ind = ga.execute_ga_nn(inputs, labels, 'catch')
np.save("model_test_catch.npy", best_ind.neural_network.parameters)
print('Time elapsed: ', (time.time() - start))

data = list(np.load('data_fifa_4actions_RGB.npy'))
inputs, labels = preprocessing_data(data, 4, 3)

print(inputs.shape)
print(labels.shape)

start = time.time()
ga = GeneticAlgorithm(10, 10, 80, 10, 20)
best_ind = ga.execute_ga_nn(inputs, labels, 'fifa')
np.save("model_test_fifa.npy", best_ind.neural_network.parameters)
print('Time elapsed: ', (time.time() - start))

data = list(np.load('training_data_snake_game_rgb.npy'))
inputs, labels = preprocessing_data(data, 4, 3)

print(inputs.shape)
print(labels.shape)

start = time.time()
ga = GeneticAlgorithm(10, 10, 80, 10, 20)
best_ind = ga.execute_ga_nn(inputs, labels, 'snake')
np.save("model_test_snake.npy", best_ind.neural_network.parameters)
print('Time elapsed: ', (time.time() - start))'''

'''param = np.load('model_test.npy')
mlp = MLP()
mlp.load_parameters(param)
print(mlp.parameters)'''