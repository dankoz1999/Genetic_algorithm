import random as rd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygad
from numpy.typing import NDArray
from datetime import datetime

SEED = 10 #random seed
NUMBER = 10  # number of items
WEIGHT = 15 #max weight
VALUE = 750 #max value
THRESHOLD = 45 #knapsack capacity


ITERATIONS = 500
COUPLES = 10
MUTATION_RATE = 0.05


class GeneticAlgo:
    def __init__(
        self,
        number: int,
        weight: int,
        value: int,
        threshold: int,
        iterations: int,
        couples: int,
        mutation_rate: float,
        additional_info: bool = False,
    ) -> None:
        self.number = number
        self.weight = weight
        self.threshold = threshold
        self.iterations = iterations
        self.couples = couples
        self.mutation_rate = mutation_rate
        self.additional_info = additional_info
        np.random.seed(SEED)
        self.knapsack_weight = np.random.randint(1, weight, size=number)
        self.knapsack_value = np.random.randint(10, value, size=number)

    def run_algo(self):
        # Zdefiniowanie problemu - wylosowanie plecaka na podstawie warunków początkowych
        if self.additional_info:
            print(f"Capacity of backpack: {self.threshold}")
            print("List of items with weight and value")
            for i in range(len(self.knapsack_weight)):
                print(
                    f"Item {i+1}: weight {self.knapsack_weight[i]}, value {self.knapsack_value[i]}"
                )
        # Zdefiniowanie populacji początkowej
        population = self._initial_population
        # Zdefiniowanie list w których zapisane zostaną średnie i najlepsze wyniki
        self.fitness_mean_algo = []
        self.fitness_best_algo = []
        self.history = []
        # Zdefiniowanie punktu przecięcia w one point crossoverze
        point = np.random.randint(1, population.shape[1], size=1)[0]
        # Wykonanie całego algorytmu przez daną liczbę iteracji
        for _ in range(self.iterations):
            # Wyliczenie wartości obj function (maksymalizujemy)
            obj = self._objective_function(population)
            # Dodanie wartości największej oraz średniej do zdefiniowanych list
            self.fitness_mean_algo.append(np.mean(obj))
            self.fitness_best_algo.append(max(obj))
            # Selekcja turniejowa
            parents = self._selection(obj, population)
            # One point crosover
            offspring = self._crossover(parents, point)
            # Mutacja z prawdopodobieństwem mutation_rate
            population = self._mutation(offspring)
        max_value = max(self.fitness_best_algo)
        index = self.fitness_best_algo.index(max_value)
        # print(f"Max value by algo -> {max_value}")
        # print(f"Choosen items by algo -> {self.history[index]}")
        return max_value, self.history[index]

    def draw(self):
        # Rysowanie wykresów Meanfitness and Maxfitness w zależności od liczby iteracji
        plt.plot(
            list(range(self.iterations)), self.fitness_mean_algo, label="Mean Fitness"
        )
        plt.plot(
            list(range(self.iterations)), self.fitness_best_algo, label="Max Fitness"
        )
        plt.legend()
        plt.title("Fitness through the generations")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.show()
        return None

    @property
    def _initial_population(self) -> NDArray:
        solutions_per_pop = self.couples * 2
        pop_size = (solutions_per_pop, self.number)
        initial_population = np.random.randint(2, size=pop_size)
        initial_population = initial_population.astype(int)
        return initial_population

    def _objective_function(self, population: NDArray) -> NDArray:
        objective = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            S1 = np.sum(population[i] * self.knapsack_value)
            S2 = np.sum(population[i] * self.knapsack_weight)
            if S2 <= self.threshold:
                objective[i] = S1
        self.history.append(population[np.argmax(objective)])
        return objective.astype(int)

    def _selection(self, fitness_function: NDArray, population: NDArray) -> NDArray:
        parents = np.empty(population.shape).astype(int)
        for i in range(fitness_function.size):
            decision = np.random.randint(fitness_function.size, size=2)
            if fitness_function[decision[0]] <= fitness_function[decision[1]]:
                parents[i] = population[decision[1]]
            else:
                parents[i] = population[decision[0]]
        return parents

    def _crossover(self, parents: NDArray, point: int) -> NDArray:
        offspring = np.empty((parents.shape)).astype(int)
        for i in range(int(parents.shape[0])):
            if i % 2 == 0:
                offspring[i][0:point] = parents[i + 1][0:point]
                offspring[i][point:] = parents[i][point:]
            else:
                offspring[i][0:point] = parents[i - 1][0:point]
                offspring[i][point:] = parents[i][point:]
        return offspring

    def _mutation(self, offspring: NDArray) -> NDArray:
        flat = offspring.flatten()
        for i in range(len(flat)):
            if rd.uniform(0, 1) <= self.mutation_rate:
                flat[i] = 1 - flat[i]
        return np.reshape(flat, offspring.shape)


# Algo from the package


def fitness_func(solution, soultion_idx):
    np.random.seed(SEED)
    knapsack_weight = np.random.randint(1, WEIGHT, NUMBER)
    knapsack_value = np.random.randint(10, VALUE, NUMBER)
    output = 0
    S1 = np.sum(solution * knapsack_value)
    S2 = np.sum(solution * knapsack_weight)
    if S2 <= THRESHOLD:
        output = S1
    return output


def run_pygad():
    fitness_function = fitness_func
    ga_instance = pygad.GA(
        ITERATIONS,
        COUPLES,
        fitness_function,
        ga._initial_population,
        init_range_low=0,
        init_range_high=1,
        gene_space=[0, 1],
    )
    ga_instance.run()
    # ga_instance.plot_fitness()
    solution, solution_fitness, _ = ga_instance.best_solution()
    return solution, solution_fitness


if __name__ == "__main__":
    start = datetime.now()
    print(f"Algorithm started @ {start}")
    df = pd.DataFrame(
        {"max_value_algo": [], "max_value_lib": [], "item_algo": [], "item_lib": []}
    )
    for i in range(100):
        SEED = i
        ga = GeneticAlgo(
            NUMBER, WEIGHT, VALUE, THRESHOLD, ITERATIONS, COUPLES, MUTATION_RATE
        )
        max_value_algo, item_algo = ga.run_algo()
        item_lib, max_value_lib = run_pygad()
        item_lib = item_lib.astype(int)
        df_temp = pd.DataFrame(
            {
                "max_value_algo": [max_value_algo],
                "max_value_lib": [int(max_value_lib)],
                "item_algo": [item_algo],
                "item_lib": [item_lib],
            }
        )
        df = pd.concat([df, df_temp])
        # print(f"Max value by lib -> {int(solution_fitness)}")
        # print(f"Chosen items by lib -> {solution.astype(int)}")
        # ga.draw()
    end = datetime.now()
    print("Dataframe Contents ", df, sep="\n")
    print(f"Algorithm finished @ {end}")
    print(f"Calculation took: {end-start}")
