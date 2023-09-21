import numpy as np
from typing import List

class Functions:
    def __init__(self) -> None:
        pass

    def Sphere(self, list : List[float]) -> float:
        result = 0

        for x in list:
            result += x*x

        return result

    def Ackley(self, list : List[float], a : float, b : float, c : float) -> float:
        result = 0



        return result

    def Rastrigin(self):
        pass

    def Rosenbrock(self):
        pass

    def Griewank(self):
        pass

    def Schwefel(self):
        pass

    def Levy(self):
        pass

    def Michalewicz(self):
        pass

    def Zakharov(self):
        pass

def BlindSearch():
    pass


if __name__ == "__main__":
    functions = Functions()
    vals = [-1.25, 10.5, 22.19, 1.59753, -2.468, 0.1379]

    print(functions.Sphere(vals))

