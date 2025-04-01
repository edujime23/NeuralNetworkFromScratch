import numpy as np
import itertools
from typing import Optional, Tuple, Callable, List, Self
from multiprocessing import Pool, cpu_count

class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation_function: Optional[Callable[[np.float64], np.float64]]):
        self.weights = np.random.randn(num_neurons, num_inputs).astype(np.float64) * np.sqrt(2.0 / num_inputs)
        self.biases = np.random.randn(num_neurons).astype(np.float64) * np.sqrt(2.0 / num_inputs)
        self.activation_function = activation_function or (lambda x: x)
        self.num_neurons = num_neurons

    def activate(self, inputs: np.ndarray, threshold: np.float64) -> np.ndarray:
        net_inputs = np.matmul(self.weights, inputs) + self.biases  # Vectorized computation
        return self.activation_function(net_inputs / threshold)  

class EvolutionaryNeuralNetwork:
    def sigmoid(self, x: np.float64) -> np.float64:
        return np.float64(1) / (np.float64(1) + np.exp(-np.clip(x, -500, 500)))

    def error(self, outputs: np.ndarray, expected: np.ndarray) -> np.float64:
        return np.sum((outputs - expected) ** 2)

    def __init__(self, input_size: int, hidden_size: Tuple[int, ...], output_size: int, threshold: float,
                 activation_functions: Optional[Tuple[Callable[[np.float64], np.float64], ...]] = None,
                 cost_function: Optional[Callable[[np.ndarray, np.ndarray], np.float64]] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.threshold = np.float64(threshold)
        self.cost = cost_function or self.error

        if activation_functions is not None:
            activation_functions = tuple(self.sigmoid if func is None else func for func in activation_functions)
            self.activation_functions = itertools.cycle(activation_functions)
        else:
            self.activation_functions = itertools.cycle([self.sigmoid])

        self.layers: List[Layer] = []
        sizes = [self.input_size] + list(self.hidden_size) + [self.output_size]
        self.layers.extend(
            Layer(sizes[i + 1], sizes[i], next(self.activation_functions))
            for i in range(len(sizes) - 1)
        )
    
    def activate(self, inputs: Tuple[float]) -> np.ndarray:
        inputs = np.array(inputs, dtype=np.float64)
        for layer in self.layers:
            inputs = layer.activate(inputs, self.threshold)  # Fully vectorized forward pass
        return inputs

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
            layer_clone.weights = np.copy(layer_self.weights)
            layer_clone.biases = np.copy(layer_self.biases)
        return cloned

    
    def crossover(self, parent2: Self) -> Self:
        """Create a child network by combining properties of two parent networks."""
        child = self.clone()

        # Randomly choose which parent's weights and biases to take
        for i in range(len(self.layers)):
            layer1 = self.layers[i]
            layer2 = parent2.layers[i]
            child_layer = child.layers[i]

            # Perform crossover on weights
            crossover_point = np.random.randint(0, layer1.weights.shape[1])
            child_layer.weights[:,:crossover_point] = layer1.weights[:,:crossover_point]
            child_layer.weights[:,crossover_point:] = layer2.weights[:,crossover_point:]

            # Perform crossover on biases (you can modify this logic if you want more control)
            child_layer.biases = layer1.biases if np.random.choice([0,1]) else layer2.biases

        return child
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1, learning_rate: float = 0.01) -> None:
        """
        Randomly perturb the network's weights and biases.
        Each parameter has a chance of mutation_rate to be adjusted by a small Gaussian noise.
        """
        for layer in self.layers:
            # Mutate weights
            for i in range(layer.weights.shape[0]):  # Iterate over neurons
                for j in range(layer.weights.shape[1]):  # Iterate over inputs to neurons
                    if np.random.rand() < mutation_rate:
                        layer.weights[i, j] += np.random.normal(0, mutation_strength) * learning_rate

            # Mutate biases
            for i in range(layer.biases.shape[0]):  # Iterate over biases
                if np.random.rand() < mutation_rate:
                    layer.biases[i] += np.random.normal(0, mutation_strength) * learning_rate

    def copy_from(self, other: Self) -> None:
        for self_layer, other_layer in zip(self.layers, other.layers):
            # Copy weights and biases directly
            self_layer.weights = np.copy(other_layer.weights)
            self_layer.biases = np.copy(other_layer.biases)


    def evaluate_fitness(self, 
                         inputs: List[Tuple[float, ...]], 
                         outputs: List[Tuple[float, ...]]) -> np.float64:
        total_error = []
        for _ in range(3):
            for input_data, output_data in zip(inputs, outputs):
                prediction = self.activate(input_data)
                total_error.append(self.cost(prediction, list(map(np.float64, output_data))))
        return np.mean(total_error)  # Return the total error without normalization

    @staticmethod
    def _fitness_worker(args):
        net: EvolutionaryNeuralNetwork = args[0]
        fitness = net.evaluate_fitness(args[1], args[2])
        return (net, fitness)
    
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

        for gen in range(generations):
            args = [(net, inputs, outputs) for net in population]

            fitness_scores = pool.map_async(EvolutionaryNeuralNetwork._fitness_worker, args)
            fitness_scores.wait()
            fitness_scores = fitness_scores.get()

            fitness_scores.sort(key=lambda x: x[1])

            population = [i[0] for i in fitness_scores]
            current_best_fitness = fitness_scores[0]

            print(f"Generation {gen}: Best fitness = {current_best_fitness[1]}")

            if best_fitness is None or current_best_fitness[1] < best_fitness[1]:
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
    hidden_size = (24,)
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
