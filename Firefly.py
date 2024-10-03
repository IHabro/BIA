import numpy as np

from functions import Functions


class Firefly:
    def __init__(self, dimension: int, boundaries: list[float], function: Functions, absorption: float, attractiveness: float):
        self.dimension = dimension
        self.boundaries = boundaries
        self.function = function

        self.position = [np.random.uniform(boundaries[0], boundaries[1]) for i in range(dimension)]
        self.value = self.function.start(self.position)

        self.lightIntensity = self.function.start(self.position)
        self.absorption = absorption
        self.attractivity = attractiveness

    def update_intensity(self, r: float):
        self.lightIntensity = self.lightIntensity * np.e**(-self.absorption * r)

    def calculate_attractiveness_better(self, r: float) -> float:
        if r > 0:
            return self.attractivity / (1.0 + r)
        elif r == 0:
            return 1.0

    def calculate_attractiveness(self, r: float) -> float:
        if r > 0:
            return self.attractivity * np.e**(-self.absorption * r * r)
        elif r == 0:
            return 1.0

    def distance(self, f):
        sm = 0.0

        if isinstance(f, Firefly):
            for x, y in zip(self.position, f.position):
                sm += (x - y) ** 2

        return np.sqrt(sm)

    def move(self, f):
        if isinstance(f, Firefly):
            r = self.distance(f)
            a = np.random.uniform(0, 1)
            e = [np.random.uniform(0, 1) for i in range(self.dimension)]

            for i in range(self.dimension):
                self.position[i] = self.position[i] + self.calculate_attractiveness_better(r) * (f.position[i] - self.position[i]) + a * e[i]

                if self.position[i] < self.boundaries[0] or self.position[i] > self.boundaries[1]:
                    self.position[i] = np.random.uniform(self.boundaries[0], self.boundaries[1])

            self.value = self.function.start(self.position)

    def move_best(self):
        new_position = self.position.copy()
        a = np.random.uniform(0, 1)
        e = [np.random.uniform(0, 1) for i in range(self.dimension)]

        for i in range(self.dimension):
            new_position[i] = self.position[i] + a * e[i]

            if new_position[i] < self.boundaries[0] or new_position[i] > self.boundaries[1]:
                new_position[i] = np.random.uniform(self.boundaries[0], self.boundaries[1])

        new_value = self.function.start(new_position)

        if new_value < self.value:
            self.value = new_value
            self.position = new_position
