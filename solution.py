import random
import numpy as np
import matplotlib.pyplot as plt

from math import e
from statistics import mean
from functions import Functions
from PSO import Swarm
from SOMA import SOMA
from Firefly import Firefly


class Solution:
    def __init__(self, dimension: int, boundaries: list[float], function: str, maxOFE: int = 3000):
        self.dimension = dimension
        self.lower = boundaries[0] # we will use the same bounds for all parameters
        self.upper = boundaries[1]
        self.parms = np.zeros(self.dimension)  # solution parameters
        self.function = Functions(function)

        self.maxOFE = maxOFE
        self.curOFE = 0

    def plot(self, points: list[list[float]], best: list[float]):
        # Inicializace grafu
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        corX = np.linspace(self.lower, self.upper)
        corY = np.linspace(self.lower, self.upper)
        corX, corY = np.meshgrid(corX, corY)
        corZ = self.function.start([corX, corY])

        # Vykresleni plochy funkce
        surf = ax.plot_surface(corX, corY, corZ, cmap="plasma", linewidth=0, antialiased=False, alpha=0.6)

        # Pridani bodu
        if self.dimension == 2:
            for point in points:
                pointToPlot = ax.scatter(point[0], point[1], self.function.start(point), c="green")

            ax.scatter(best[0], best[1], self.function.start(best), c="red")

        # Vykresleni vysledku
        plt.title = self.function.name
        plt.show()

    def get_empty_by_dimension(self) -> list[float]:
        return [0.0 for i in range(self.dimension)]

    def get_random_from_boundaries(self) -> list[float]:
        result = []

        for i in range(self.dimension):
            result.append(random.uniform(self.lower, self.upper))

        return result

    def get_normal_from_solution(self, bestSolution: list[float], sigma: float) -> list[float]:
        result = []

        for i in range(self.dimension):
            result.append(np.random.normal(bestSolution[i], sigma, 1))

        return result

    def get_normal_from_boundaries(self, sigma: float) -> list[float]:
        result = []
        boundaries = [self.lower, self.upper]

        for x in boundaries:
            result.append(np.random.uniform(x, sigma, 1).tolist())

        return result

    # Implementace BlindSearch algoritmu
    def blind_search(self, gens: int, points: int):
        bestSolution = self.get_random_from_boundaries()
        bestValue = self.function.start(bestSolution)
        bestSolutions = []

        genSolutions = []
        values = []

        for g in range(gens):
            for p in range(points):
                solution = self.get_random_from_boundaries()
                genSolutions.append(solution)
                values.append(self.function.start(solution))

            mn = min(values)

            if mn < bestValue:
                bestSolution = genSolutions[values.index(mn)]
                bestValue = mn
                bestSolutions.append(bestSolution)

        self.plot(bestSolutions, bestSolution)

    def hill_climb(self, gens: int, points: int, sigma: float):
        bestSolution = self.get_random_from_boundaries()
        bestValue = self.function.start(bestSolution)
        bestSolutions = []

        genSolutions = []
        values = []

        for g in range(gens):
            for p in range(points):
                solution = self.get_normal_from_solution(bestSolution, sigma)
                genSolutions.append(solution)
                values.append(self.function.start(solution))

            mn = min(values)

            if mn < bestValue:
                bestSolution = genSolutions[values.index(mn)]
                bestValue = mn
                bestSolutions.append(bestSolution)

        self.plot(bestSolutions, bestSolution)

    # Blind search starting from self.upper, self.upper
    def test_blind(self, gens: int, points: int):
        bestSolution = [self.upper, self.upper]
        bestValue = self.function.start(bestSolution)
        bestSolutions = []
        bestSolutions.append(bestSolution)

        genSolutions = []
        values = []

        if gens <= 0 or points <= 0:
            return bestSolutions

        for g in range(gens):
            for p in range(points):
                solution = self.get_random_from_boundaries()
                genSolutions.append(solution)
                values.append(self.function.start(solution))

            mn = min(values)
            bestSolutions.append(genSolutions[values.index(mn)])

            if mn < bestValue:
                bestSolution = genSolutions[values.index(mn)]
                bestValue = mn

        return bestSolutions

    def simulate_annealing(self, temp: float, lowTemp: float, alpha: float, sigma: float):
        bestSolution = self.get_random_from_boundaries()
        bestValue = self.function.start(bestSolution)
        bestSolutions = []

        while temp > lowTemp:
            solution = self.get_normal_from_solution(bestSolution, sigma)
            value = self.function.start(solution)

            if value < bestValue:
                bestSolution = solution
                bestValue = value

                bestSolutions.append(bestSolution)
            else:
                r = self.get_normal_from_boundaries(sigma)
                exp = (value - bestValue)/temp

                if self.function.start(r) < e**(-1 * exp):
                    bestSolution = solution
                    bestValue = value

                    bestSolutions.append(bestSolution)

            temp = temp*alpha

        self.plot(bestSolutions, bestSolution)

    def create_mutation_vector(self, parent1: list[float], parent2: list[float], parent3: list[float], multiple: float) -> list[float]:
        result = []

        for i in range(len(parent1)):
            value = (parent1[i] - parent2[i]) * multiple + parent3[i]

            # Pokud utekl, tak vem nahodny z normalniho rozlozeni
            if value < self.lower or value > self.upper:
                value = random.uniform(self.lower, self.upper)

            result.append(value)

        return result

    def differential_evolution(self, pops: int, gens: int, mutationChance: float = 0.5, crossOverRange: float = 0.5) -> float:
        population = [self.get_random_from_boundaries() for p in range(pops)]
        evaluation = [self.function.start(p) for p in population]

        for p in population:
            if self.curOFE <= self.maxOFE:
                evaluation.append(self.function.start(p))
                self.curOFE += 1

        bestSolutions = [population[evaluation.index(min(evaluation))]]

        for g in range(gens):
            new_population = population.copy()

            for i, x in enumerate(population):
                data = list(range(pops))
                data.remove(i)
                r1, r2, r3 = np.random.choice(data, 3, replace=False)

                vector = self.create_mutation_vector(population[r1], population[r2], population[r3], mutationChance)
                child = self.get_empty_by_dimension()
                j_rnd = np.random.randint(0, self.dimension)

                for j in range(self.dimension):
                    if np.random.uniform(0, 1) < crossOverRange or j == j_rnd:
                        child[j] = vector[j]
                    else:
                        child[j] = x[j]

                eval_child = -1

                if self.curOFE <= self.maxOFE:
                    eval_child = self.function.start(child)
                    self.curOFE += 1
                else:
                    self.curOFE = 0
                    return min(evaluation)

                if eval_child <= evaluation[i]:
                    new_population[i] = child
                    evaluation[i] = eval_child

            population = new_population
            # Toto nepotrebuju bo jsem v situaci ze pri zmene menim i funkcni hodnotu -> o 1 OFE min
            # evaluation = [self.function.start(p) for p in population]

            bestSolutions.append(population[evaluation.index(min(evaluation))])

        # Tady me OFE nezajimaji bo uz jsou davno vypoctene a kod by se sem nemel dostat
        bestValues = [self.function.start(val) for val in bestSolutions]
        self.plot(bestSolutions, bestSolutions[bestValues.index(min(bestValues))])

        self.curOFE = 0
        return -3.14

    def particle_swarm(self, pop_size: int, migrations: int, constants: list[float], velocities: list[float], intertia: list[float]) -> float:
        swarm = Swarm(pop_size, self.dimension, [self.lower, self.upper], constants, velocities, self.function)
        w_s, w_e = intertia

        for m in range(migrations):
            w = w_s - (((w_s - w_e) * m) / migrations)
            result = swarm.migrate(w)

            if not result == -3.14:
                return result

        self.plot(swarm.get_points(), swarm.gBest.position)

        return -3.14

    def soma(self, pop_size: int, migrations: int, prt: float = 0.4, path_length: float = 3.0, step: float = 0.11) -> float:
        soma = SOMA(pop_size, self.dimension, prt, path_length, step, [self.lower, self.upper], self.function)

        for m in range(migrations):
            result = soma.migrate()

            if not result == -3.14:
                return result

        self.plot(soma.get_points(), soma.leader.position)

        return -3.14

    def firefly(self, pops: int, gens: int, absorption: float = 0.01, attractiveness: float = 1, maxOFE: int = 3000) -> float:
        flock = [Firefly(self.dimension, [self.lower, self.upper], self.function, absorption, attractiveness) for p in range(pops)]
        evalu = [f.value for f in flock]
        best_index = evalu.index(min(evalu))
        best_solutions = [flock[best_index]]

        curOFE = pops

        # self.plot([f.position for f in flock], best_solutions[-1].position)

        for g in range(gens):
            for i in range(pops):
                for j in range(pops):

                    if curOFE <= maxOFE:
                        # Problemova podminka
                        if evalu[j] < evalu[i]:
                            flock[i].move(flock[j])
                            curOFE += 1
                            continue

                        if i == j == best_index:
                            flock[i].move_best()
                            curOFE += 1
                            continue

            evalu = [f.value for f in flock]
            best_index = evalu.index(min(evalu))

            if evalu[best_index] < best_solutions[-1].value:
                best_solutions.append(flock[best_index])

            # Vrat posledniho nejlepsiho typecka, vzhledem k tomu, ze jsem za hranici kontroly, tak vzdy vrati spravny vysledek
            if curOFE > maxOFE:
                return best_solutions[-1].value

        self.plot([f.position for f in flock], best_solutions[-1].position)

        curOFE = 0
        return -3.14
        
    def tlbo(self, pops: int, maxOFE: int = 3000) -> float:
        curOFE = 0

        # Initialization
        population = [self.get_random_from_boundaries() for i in range(pops)]
        eval = []
        for p in population:
            eval.append(self.function.start(p))
            curOFE += 1

        # Do until best is found or maxOFE is reached
        while True:
            # For each student
            for i in range(pops):
                # These guys could change over time
                teacher = eval.index(min(eval))
                M = mean(eval)  # Mean should be the mean of all student in the class -> mean of eval should suffice

                # Teacher phase
                Xnew = []
                r = random.uniform(0, 1)
                tf = np.random.randint(1, 3)

                for iter in range(self.dimension):
                    Xnew.append(population[teacher][iter] + r * (population[teacher][iter] - tf * M))   # Should I choose r and TF randomly every time?

                if curOFE <= maxOFE:
                    Enew = self.function.start(Xnew)
                    curOFE += 1
                else:
                    return min(eval)

                if Enew < eval[teacher]:
                    population[teacher] = Xnew

                # Learner phase
                # Find comrade
                j = -1
                while True:
                    j = np.random.randint(0, pops)
                    if not i == j:
                        break

                # For each dude at I and comrade at J
                for iter in range(self.dimension):
                    if eval[i] < eval[j]:
                        Xnew[iter] = population[i][iter] + r * (population[i][iter] - population[j][iter])
                    else:
                        Xnew[iter] = population[i][iter] + r * (population[j][iter] - population[i][iter])

                    if curOFE <= maxOFE:
                        Enew = self.function.start(Xnew)
                        curOFE += 1
                    else:
                        return min(eval)

                    if Enew < eval[i]:
                        population[i] = Xnew

        return min(eval)

    def comparison(self, dimension: int = 30, populationSize: int = 30, maxOFE: int = 3000) -> list[list[float]]:
        csvResult: list[list[float]] = []

        # Vsechny funkce musi za sebou uklizet a zapsat do self.curOFE nulu
        for i in range(1, dimension + 1):
            # Valim do gens maxOFE, bo potrebuju v DE nejake cislo, dle kodu by mel skoncit drive
            de = self.differential_evolution(populationSize, maxOFE)
            pso = self.particle_swarm(populationSize, maxOFE, [2.0, 2.2], [-1.0, 1.0], [0.9, 0.4])
            soma = self.soma(populationSize, maxOFE)
            fa = self.firefly(populationSize, maxOFE)
            tlbo = self.tlbo(populationSize, maxOFE)

            csvResult.append([de, pso, soma, fa, tlbo])

        # print(self.function.name)
        # print("DE, PSO, SOMA, FA, TLBO")
        # for i in range(len(csvResult)):
        #     print("{} {}".format(i + 1, csvResult[i]))

        return csvResult
