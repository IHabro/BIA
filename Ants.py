import numpy as np
from numpy import cumsum, random
from salesman import City
import matplotlib.pyplot as plt
from time import sleep
import copy


class Colony:
    def __init__(self, info: list[int], constants: list[float], boundaries: list[int], names: list[str]):
        self.population_size = info[0]
        self.antingCycles = info[1]
        self.numberOfCities = info[2]
        self.alpha = constants[0]
        self.beta = constants[1]
        self.evaporation = constants[2]

        self.names = names
        self.boundaries = boundaries

        self.cities = self.generate_cities()
        self.result = [np.inf]

        self.distances = [[0.0, 10.0, 12.0, 11.0, 14.0],
                          [10.0, 0.0, 13.0, 15.0, 8.0],
                          [12.0, 13.0, 0.0, 9.0, 4.0],
                          [11.0, 15.0, 9.0, 0.0, 16.0],
                          [14.0, 8.0, 14.0, 16.0, 0.0]]
        self.visibility = [[0.0, 0.1000, 0.0833, 0.0909, 0.0174],
                           [0.1000, 0.0, 0.0769, 0.0667, 0.1250],
                           [0.0833, 0.0769, 0.0, 0.1111, 0.0714],
                           [0.0909, 0.0667, 0.1111, 0.0, 0.0625],
                           [0.0714, 0.125, 0.0714, 0.0625, 0.0]]
        self.pheromones = [[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]]

        self.fill_matrixes()

        self.ants = []
        self.generate_ants()

    def generate_cities(self) -> list[City]:
        return [City(self.names[i], self.boundaries) for i in range(len(self.names))]

    def generate_ants(self):
        self.ants.clear()

        for a in range(self.population_size):
            ant = Ant(self.numberOfCities, [self.alpha, self.beta], copy.deepcopy(self.distances),
                      copy.deepcopy(self.visibility), copy.deepcopy(self.pheromones))
            self.ants.append(ant)

    def fill_matrixes(self):
        self.distances.clear()
        self.visibility.clear()
        self.pheromones.clear()

        for r in range(self.numberOfCities):
            distnaces = []
            visibility = []
            pheromones = []

            for c in range(self.numberOfCities):
                tmp = np.sqrt((self.cities[r].coordinates[0] - self.cities[c].coordinates[0]) ** 2
                              + (self.cities[r].coordinates[1] - self.cities[c].coordinates[1]) ** 2)
                distnaces.append(round(tmp, 4))
                visibility.append(round(1.0 / tmp, 4) if not tmp == 0 else 0.0)
                pheromones.append(1)

            self.distances.append(distnaces)
            self.visibility.append(visibility)
            self.pheromones.append(pheromones)

    def recalculate_pheromones(self):
        for r in range(self.numberOfCities):
            for c in range(self.numberOfCities):
                self.pheromones[r][c] = (1 - self.evaporation)*self.pheromones[r][c]

        for a in self.ants:
            value = 1.0/a.pathCost
            path = a.path

            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i+1]] += value

    def anting(self):
        for c in range(self.antingCycles):
            for a in self.ants:
                a.anting()

            self.recalculate_pheromones()
            self.plot_best_ant("{} cyklus".format(c))

            if not c == self.antingCycles - 1:
                self.generate_ants()

        self.plot_best_ant("Konecny vysledek")
        print(self.result)

    def plot_best_ant(self, name: str):
        cycleResult = [a.pathCost for a in self.ants]
        mn = min(cycleResult)

        if mn < self.result[-1]:
            indx = cycleResult.index(mn)
            ant = self.ants[indx]

            x = []
            y = []

            for i in ant.path:
                x.append(self.cities[i].coordinates[0])
                y.append(self.cities[i].coordinates[1])

            plt.title(name)
            plt.scatter(x, y)
            plt.plot(x, y)
            plt.show()
            sleep(1)

            self.result.append(ant.pathCost)


class Ant:
    def __init__(self, number: int, constants: list[float], distnaces: list[list[float]], visibility: list[list[float]], pheromones: list[list[float]]):
        # Step 1
        self.startingCity: int = random.randint(0, number - 1)
        self.numberOfCities = number
        self.alpha = constants[0]
        self.beta = constants[1]
        self.path = [self.startingCity]
        self.pathCost = 0.0
        self.cityIndexes = [i for i in range(self.numberOfCities)]
        self.distances = distnaces
        self.visibility = visibility
        self.pheromones = pheromones

    def set_matrixes(self, visibility: list[list[float]], pheromones: list[list[float]]):
        self.visibility = visibility
        self.pheromones = pheromones

        self.path = [self.startingCity]
        self.pathCost = 0.0
        self.cityIndexes = [i for i in range(self.numberOfCities)]

    def anting(self):
        self.clear_visibility(self.startingCity)
        self.walk_the_path()
        self.calculate_distance()

        print(self.path)
        print(self.pathCost)

    def calculate_distance(self):
        for i in range(self.numberOfCities - 1):
            self.pathCost += self.distances[self.path[i]][self.path[i + 1]]

        self.pathCost = round(self.pathCost, 4)

    def walk_the_path(self):
        probabilities = 0.0
        nextCity = -1
        for i in range(self.numberOfCities - 1):
            if i == 0:
                probabilities = self.calculate_probabilities(self.startingCity)
            else:
                probabilities = self.calculate_probabilities(nextCity)
            nextCity = self.get_next_city(probabilities)
            self.path.append(nextCity)

        # Cesta zase zpatky
        self.path.append(self.startingCity)

    def get_next_city(self, probabilities: list[float]) -> int:
        r = random.uniform(0, 1)

        for i in self.cityIndexes:
            if probabilities[i - 1] <= r <= probabilities[i]:
                return self.clear_visibility(i)
            elif i == 0:
                if 0 <= r < probabilities[i]:
                    return self.clear_visibility(i)

        return -1

    def clear_visibility(self, index: int) -> int:
        for i in range(self.numberOfCities):
            self.visibility[i][index] = 0.0

        return index

    # Step 2 requires input between 0 and self.numberOfCiteis - 1
    def calculate_probabilities(self, startingCity: int) -> list[float]:
        self.cityIndexes.remove(startingCity)
        result = []
        tmpResult = []
        sm = 0.0

        for i in range(self.numberOfCities):
            tmp = self.pheromones[startingCity][i]**self.alpha * self.visibility[startingCity][i]**self.beta
            tmpResult.append(tmp)
            sm += tmp

        for i in range(self.numberOfCities):
            tmp = tmpResult[i]/sm
            result.append(tmp)

        result = cumsum(result)

        return list(result)
