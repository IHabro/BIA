import random
from functions import Functions


class Individual:
    def __init__(self, dimension: int, boundaries: list[float], function: Functions, path_length: float, step: float, prt: float):
        self.dimension = dimension
        self.boundaries = boundaries
        self.function = function

        self.position = [random.uniform(boundaries[0], boundaries[1]) for i in range(dimension)]
        self.value = self.function.start(self.position)

        self.path_length = path_length
        self.step = step
        self.prt = prt

    # Traveling individual
    def recalculate(self, leader, maxOFE: int = 3000, curOFE: int = 0):
        if isinstance(leader, Individual):
            # PRT Vector before each calculation
            prt_vector = self.generate_vector()
            positions = []

            # Until t > path_length
            t = 0.0
            while t <= self.path_length:
                result = [0 for i in range(self.dimension)]

                # Add new possible positions to the list
                for i in range(self.dimension):
                    result[i] = self.position[i] + (leader.position[i] - self.position[i]) * t * prt_vector[i]

                    if result[i] < self.boundaries[0] or result[i] > self.boundaries[1]:
                        result[i] = random.uniform(self.boundaries[0], self.boundaries[1])

                positions.append(result)
                t = t + self.step

            # Evaluate possible positions
            evaluation = []
            for p in positions:
                if curOFE <= maxOFE:
                    evaluation.append(self.function.start(p))
                    curOFE += 1
                else:
                    return 3.14, curOFE

            # If best from positions is better then current -> replace current
            min_val = min(evaluation)
            if min_val < self.value:
                self.value = min_val
                self.position = positions[evaluation.index(min_val)]

        return -3.14, curOFE

    def generate_vector(self) -> list[float]:
        return [1.0 if random.uniform(0, 1) < self.prt else 0.0 for i in range(self.dimension)]


class SOMA:
    def __init__(self, pops: int, dimension: int, prt: float, path_length: float, step: float, boundaries: list[float],
                 function: Functions, maxOFE: int = 3000):
        self.population = [Individual(dimension, boundaries, function, path_length, step, prt) for i in range(pops)]
        self.evaluation = [p.value for p in self.population]
        self.leader = self.population[self.evaluation.index(min(self.evaluation))]

        self.prt = prt
        self.pops = pops

        # First OFE inside the individual
        self.maxOFE = maxOFE
        self.curOFE = pops

    def migrate(self) -> float:
        for p in self.population:
            if not p == self.leader:
                result, self.curOFE = p.recalculate(self.leader, self.maxOFE, self.curOFE)

                if result == 3.14:
                    return min([i.value for i in self.population])

        return -3.14

    def get_points(self) -> list[list[float]]:
        return [p.position for p in self.population]
