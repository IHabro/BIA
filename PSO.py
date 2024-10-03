import random
from functions import Functions


class Swarmling:
    def __init__(self, dimension: int, boundaries: list[float], constants: list[float], velocities: list[float], function: Functions):
        self.dimension = dimension
        self.function = function
        self.constants = constants
        self.velocity_boundaries = velocities
        self.data_boundaries = boundaries

        self.position = [random.uniform(boundaries[0], boundaries[1]) for i in range(dimension)]
        self.velocities = [0.0 for i in range(dimension)]
        self.pBest = self.position
        self.value = self.function.start(self.position)
        self.old_value = self.value

    def calculate_new_position(self, gBest, inertia: float):
        if isinstance(gBest, Swarmling):
            c1, c2 = self.constants

            for i in range(self.dimension):
                r = random.uniform(0, 1)
                # v(t) -> v(t+1)
                self.velocities[i] = self.velocities[i]*inertia + c1*r*(self.pBest[i] - self.position[i]) + c2*r*(gBest.position[i] - self.position[i])
                if self.velocities[i] < self.velocity_boundaries[0] or self.velocities[i] > self.velocity_boundaries[1]:
                    self.velocities[i] = random.uniform(self.velocity_boundaries[0], self.velocity_boundaries[1])

                # x(t) -> x(t+1)
                self.position[i] = self.position[i] + self.velocities[i]
                if self.position[i] < self.data_boundaries[0] or self.position[i] > self.data_boundaries[1]:
                    self.position[i] = random.uniform(self.data_boundaries[0], self.data_boundaries[1])

            self.value = self.function.start(self.position)

            if self.value < self.old_value:
                self.pBest = self.position
                self.old_value = self.value


class Swarm:
    def __init__(self, swarmlings: int, dimension: int, boundaries: list[float], learning: list[float], velocities: list[float],
                 function: Functions, maxOFE: int = 3000):
        self.swarmlings = [Swarmling(dimension, boundaries, learning, velocities, function) for i in range(swarmlings)]
        self.pBests = [s.pBest for s in self.swarmlings]
        self.gBest = self.swarmlings[self.pBests.index(min(self.pBests))]

        # First OFE inside the individual
        self.maxOFE = maxOFE
        self.currOFE = swarmlings

    def migrate(self, inertia: float) -> float:
        for i, s in enumerate(self.swarmlings):
            if self.currOFE <= self.maxOFE:
                s.calculate_new_position(self.gBest, inertia)
                self.currOFE += 1
            else:
                self.currOFE = 0
                return min([s.value for s in self.swarmlings])

            if s.value < self.gBest.value:
                self.gBest = s

        return -3.14

    def get_points(self) -> list[list[float]]:
        return [s.position for s in self.swarmlings]
