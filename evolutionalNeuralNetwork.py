import numpy as np
import itertools
from typing import Optional, Tuple, Callable, List, Self
from multiprocessing import Pool, cpu_count, set_start_method

class Neuron:
    def __init__(self, num_inputs: int, activation_function: Callable[[np.float64], np.float64]):
        self.delta: np.float64 = np.float64(0)
        self.weights: np.ndarray = np.random.randn(num_inputs).astype(np.float64) * np.sqrt(2.0 / num_inputs)
        self.bias: np.float64 = np.random.randn(1)[0] * np.sqrt(2.0 / num_inputs)
        self.signal: np.float64 = np.float64(0)
        self.activation_function = activation_function

    def activate(self, inputs: np.ndarray, threshold: np.float64) -> None:
        signal_sum: np.float64 = self.bias + np.dot(self.weights, inputs.astype(np.float64))  
        self.signal = self.activation_function(signal_sum / threshold)  

class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation_function: Optional[Callable[[np.float64], np.float64]]):
        self.neurons: List[Neuron] = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]  
        self.activation_function = activation_function  
        self.num_neurons = num_neurons  
        self.num_inputs = num_inputs  

class EvolutionaryNeuralNetwork:
    def sigmoid(self, x: np.float64) -> np.float64:
        return np.float64(1) / (np.float64(1) + np.exp(-np.clip(x, -500, 500)))

    def error(self, outputs: List[np.float64], expected: List[np.float64]) -> np.float64:
        outputs = np.array(outputs, dtype=np.float64)
        expected = np.array(expected, dtype=np.float64)
        mean = np.sum((outputs - expected)**2)
        
        return mean
    
    def derivative(self, activation: Callable[[np.float64], np.float64], x: np.float64, *, dx: np.float64 = np.float64(10e-8)) -> np.float64:
        a1 = activation(x + dx)
        a2 = activation(x)
        return (a1 - a2) / dx

    def __init__(self, 
                 input_size: int, 
                 hidden_size: Tuple[int, ...], 
                 output_size: int, 
                 threshold: float,
                 activation_functions: Optional[Tuple[Callable[[np.float64], np.float64], ...]] = None,
                 cost_function: Optional[Callable[[List[np.float64], List[np.float64]], np.float64]] = None):
        if not isinstance(hidden_size, tuple):
            raise TypeError("hidden_size must be a tuple of ints")
        if not all(isinstance(x, int) for x in hidden_size):
            raise TypeError("All elements in hidden_size must be ints")
        if not isinstance(input_size, int) or not isinstance(output_size, int):
            raise TypeError("input_size and output_size must be ints")
        if input_size <= 0 or output_size <= 0 or any(x <= 0 for x in hidden_size):
            raise ValueError("input_size, output_size, and all elements in hidden_size must be positive integers")
        
        if activation_functions is not None:
            activation_functions = tuple(self.sigmoid if func is None else func for func in activation_functions)
            self.activation_functions = itertools.cycle(activation_functions)
        else:
            self.activation_functions = itertools.cycle([self.sigmoid])
        
        if cost_function is not None and not callable(cost_function):
            raise TypeError("cost_function must be callable")
        self.cost = cost_function if cost_function else self.error
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = np.float64(threshold)
        
        self.layers: List[Layer] = [Layer(self.input_size, self.input_size, None)]
        for i in range(len(self.hidden_size)):
            self.layers.append(Layer(self.hidden_size[i], self.hidden_size[i-1] if i > 0 else self.input_size, next(self.activation_functions)))
        self.layers.append(Layer(self.output_size, self.hidden_size[-1] if self.hidden_size else self.input_size, next(self.activation_functions)))
    
    def activate(self, inputs: Tuple[float]) -> List[np.float64]:
        inputs = np.array(inputs, dtype=np.float64)
        for i in range(len(inputs)):
            self.layers[0].neurons[i].signal = inputs[i]
        for i in range(1, len(self.layers)):
            prev_signals = np.array([neuron.signal for neuron in self.layers[i-1].neurons], dtype=np.float64)
            for neuron in self.layers[i].neurons:
                neuron.activate(prev_signals, self.threshold)
        return [neuron.signal for neuron in self.layers[-1].neurons]

    def clone(self) -> Self:
        cloned = EvolutionaryNeuralNetwork(
            input_size=self.input_size, 
            hidden_size=tuple(int(x) for x in self.hidden_size), 
            output_size=self.output_size, 
            threshold=self.threshold,
            activation_functions=None,  
            cost_function=self.cost
        )
        for layer_self, layer_clone in zip(self.layers, cloned.layers):
            for neuron_self, neuron_clone in zip(layer_self.neurons, layer_clone.neurons):
                neuron_clone.weights = np.copy(neuron_self.weights)
                neuron_clone.bias = neuron_self.bias
        return cloned
    
    def crossover(parent1: Self, parent2: Self) -> Self:
        """Create a child network by combining properties of two parent networks."""
        child = parent1.clone()
        
        # Randomly choose which parent's weights to take
        for i in range(len(parent1.layers)):
            for j in range(len(parent1.layers[i].neurons)):
                parent1_neuron = parent1.layers[i].neurons[j]
                parent2_neuron = parent2.layers[i].neurons[j]
                child_neuron = child.layers[i].neurons[j]
                
                # Perform crossover on weights and biases
                crossover_point = np.random.randint(0, len(parent1_neuron.weights))
                child_neuron.weights[:crossover_point] = parent1_neuron.weights[:crossover_point]
                child_neuron.weights[crossover_point:] = parent2_neuron.weights[crossover_point:]
                
                # You can similarly crossover biases if desired
                child_neuron.bias = np.random.choice([parent1_neuron.bias, parent2_neuron.bias])
        
        return child
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1, learning_rate: float = 0.01) -> None:
        """
        Randomly perturb the network's weights and biases.
        Each parameter has a chance of mutation_rate to be adjusted by a small Gaussian noise.
        """
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    if np.random.rand() < mutation_rate:
                        neuron.weights[i] += np.random.normal(0, mutation_strength) * learning_rate
                if np.random.rand() < mutation_rate:
                    neuron.bias += np.random.normal(0, mutation_strength) * learning_rate

    def copy_from(self, other: Self) -> None:
        for self_layer, other_layer in zip(self.layers, other.layers):
            for self_neuron, other_neuron in zip(self_layer.neurons, other_layer.neurons):
                self_neuron.weights = np.copy(other_neuron.weights)
                self_neuron.bias = other_neuron.bias

    def evaluate_fitness(self, 
                         inputs: List[Tuple[float, ...]], 
                         outputs: List[Tuple[float, ...]]) -> np.float64:
        total_error = np.float64(0.0)
        for input_data, output_data in zip(inputs, outputs):
            prediction = self.activate(input_data)
            total_error += self.cost(prediction, list(map(np.float64, output_data)))
        return total_error  # Return the total error without normalization

    @staticmethod
    def _fitness_worker(args):
        net: EvolutionaryNeuralNetwork = args[0]
        fitness = net.evaluate_fitness(args[1], args[2])
        return fitness
    
    def train(self, 
              inputs: List[Tuple[float, ...]], 
              outputs: List[Tuple[float, ...]], 
              generations: int, 
              population_size: int, 
              elitism: float = 0.2, 
              mutation_rate: float = 0.1, 
              mutation_strength: float = 0.1,
              learning_rate: float = 0.01) -> None:
        population = [self.clone() for _ in range(population_size)]
        elite_count = max(1, int(elitism * population_size))
        
        best_fitness = None
        best_network = None
        stagnant_generations = 0
        
        pool = Pool(processes=cpu_count())
        
        for gen in range(int(generations)):
            args = [(net, inputs, outputs) for net in population]
            
            fitness_scores = pool.map_async(EvolutionaryNeuralNetwork._fitness_worker, args)
            fitness_scores.wait()
            fitness_scores = fitness_scores.get()
            
            sorted_indices = np.argsort(fitness_scores)
            population = [population[i] for i in sorted_indices]
            current_best_fitness = fitness_scores[sorted_indices[0]]
            
            print(f"Generation {gen}: Best fitness = {current_best_fitness}")
            
            if best_fitness is None or current_best_fitness < best_fitness:
                best_network = population[0].clone()
                best_fitness = current_best_fitness
                stagnant_generations = 0
            elif best_fitness == 0:
                break
            else:
                stagnant_generations += 1
            
            if stagnant_generations >= 10:
                mutation_strength /= 2
                print(f"Mutation strength adjusted to {mutation_strength}")
                stagnant_generations = 0
            
            elites = population[:elite_count]
            offspring = []
            while len(offspring) < (population_size - elite_count):
                parent1 = np.random.choice(elites)
                parent2 = np.random.choice(elites)
                child = EvolutionaryNeuralNetwork.crossover(parent1, parent2)
                child.mutate(mutation_rate, mutation_strength, learning_rate)
                offspring.append(child)
            population = elites + offspring
        
        self.copy_from(best_network)
        
        pool.close()
        pool.join()

if __name__ == "__main__":
    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_expected = [(0,), (1,), (1,), (0,)]

    input_size = 2
    hidden_size = (6,)
    output_size = 1
    threshold = 1  
    activation_functions = (np.tanh,None)  # Specify activation functions for each layer
    cost_function = None
    
    mutation_rate = 6/9
    mutation_strength = 512
    generations = 25
    population_size = 256
    elitism = 3/9  
    learning_rate = 2/3

    nn = EvolutionaryNeuralNetwork(input_size, hidden_size, output_size, threshold, activation_functions=activation_functions, cost_function=cost_function)
    nn.train(xor_inputs, xor_expected, generations=generations, population_size=population_size, elitism=elitism, mutation_rate=mutation_rate, mutation_strength=mutation_strength, learning_rate=learning_rate)

    # do a cost func test
    print("Cost function test:")
    print(nn.evaluate_fitness(xor_inputs, xor_expected))

    for i in range(1, len(nn.layers)):
        print(f"Layer {i}: {nn.layers[i].activation_function.__name__} {nn.layers[i].num_neurons} neurons")

    # Test the trained network
    print("Testing evolved network:")
    for inputs in xor_inputs:
        output = nn.activate(inputs)
        print(f"Input: {inputs}, Output: {output[0].round(7)}")
