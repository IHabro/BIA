import numpy as np
import matplotlib.pyplot as plt
from time import sleep


# Wrapper class for City
class City:
    def __init__(self, name: str, boundaries: list[int]):
        self.name = name
        self.coordinates = [np.random.randint(0, boundaries[0]), np.random.randint(0, boundaries[1])]


# Wrapper class for Individual in population
class Individual:
    def __init__(self, cities: list[City], first: list[City]):
        self.size = len(cities)
        self.cities: list[City] = cities
        self.evaluation = 0.0

        # Shuffle cities
        np.random.shuffle(self.cities)
        self.cities = first + self.cities

        self.evaluate()

    # random mutation of 2 cities with range 1-size
    def mutate(self):
        i = np.random.randint(1, self.size)
        j = -1

        while True:
            j = np.random.randint(1, self.size)
            if i != j:
                break

        self.cities[i], self.cities[j] = self.cities[j], self.cities[i]
        self.evaluate()

    # Return cities which are not included in lst
    def return_nonselected_cities(self, lst: list[City]) -> list[City]:
        mine = [c.name for c in self.cities]
        lst = [c.name for c in lst]
        result = []

        for i in range(len(mine)):
            if not lst.__contains__(mine[i]):
                result.append(self.cities[i])

        return result

    # Application of pythagorean theorem on cities
    def evaluate(self):
        for i in range(self.size - 1):
            val = np.sqrt((self.cities[i].coordinates[0] - self.cities[i + 1].coordinates[0])**2
                                       + (self.cities[i].coordinates[1] - self.cities[i + 1].coordinates[1])**2)
            self.evaluation += val

        self.evaluation += np.sqrt((self.cities[-1].coordinates[0] - self.cities[0].coordinates[0]) ** 2
                                   + (self.cities[-1].coordinates[1] - self.cities[0].coordinates[1]) ** 2)

    def get_coordinates(self) -> list[list[float]]:
        x = [c.coordinates[0] for c in self.cities]
        y = [c.coordinates[1] for c in self.cities]
        return [x, y]


class Salesman:
    def __init__(self, names: list[str], boundaries: list[int]):
        self.names = names
        self.boundaries = boundaries

    def generate_first_population(self, pops: int) -> list[Individual]:
        cities = [City(self.names[i], self.boundaries) for i in range(len(self.names))]
        first = cities.pop(0)

        result = []

        for i in range(pops):
            ind = Individual(cities, [first])
            result.append(ind)

        return result

    def evaluate_population(self, population: list[Individual]) -> list[float]:
        return [round(p.evaluation, 3) for p in population]

    def select_B(self, population: list[Individual], A: int):
        i: int = -1

        while True:
            i = np.random.randint(0, len(population))

            if i != A:
                break

        return population[i]

    def create_offspring(self, A: Individual, B: Individual):
        if len(A.cities) != len(B.cities):
            return

        # Calculate middle index + concate first half of A and second half of B
        mdl = round(len(A.cities) / 2)
        first = A.cities[:mdl]
        second = B.return_nonselected_cities(first)
        cities = first + second

        first = cities.pop(0)
        result = Individual(cities, [first])

        return result

    def plot_population(self, population: list[Individual], evaluation: list[float], title: str):
        x, y = population[evaluation.index(min(evaluation))].get_coordinates()

        plt.title(title)
        plt.scatter(x, y)
        plt.plot(x + [x[0]], y + [y[0]])
        plt.show()
        sleep(1)

    def genetic_algorithm(self, pops: int, gens: int, citiesNumber: int, mutationChance: float):
        population = self.generate_first_population(pops)
        evaluation = self.evaluate_population(population)
        bestSolutions = [min(evaluation)]

        self.plot_population(population, evaluation, "First")

        print("{}-{}".format(-1, bestSolutions[-1]))

        for g in range(gens):
            new_population = population.copy()

            for i in range(pops):
                offspring = self.create_offspring(population[i], self.select_B(population, i))

                if np.random.uniform() < mutationChance:
                    offspring.mutate()

                if offspring.evaluation < population[i].evaluation:
                    new_population[i] = offspring

            population = new_population
            evaluation = self.evaluate_population(population)

            if min(evaluation) < bestSolutions[-1]:
                bestSolutions.append(min(evaluation))
                print("{}-{}".format(g, min(evaluation)))
                self.plot_population(population, evaluation, "{}".format(len(bestSolutions)-1))

        print("{}-{}".format(np.inf, min(evaluation)))
        self.plot_population(population, evaluation, "Last")
        