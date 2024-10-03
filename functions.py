from numpy import sin, sqrt, cos, pi, exp


# Implementace funkci ze zadani Tasku
class Functions:
    def __init__(self, name):
        self.name = name

    # Switch nad funkcema
    def start(self, lst: list[float]) -> float:
        if self.name == "sphere":
            return self.sphere(lst)
        elif self.name == "schwefel":
            return self.schwefel(lst)
        elif self.name == "rosenbrock":
            # return self.rosenbrock(lst)
            return self.new_rosenbrock(lst)
        elif self.name == "rastrigin":
            return self.rastrigin(lst)
        elif self.name == "griewank":
            return self.griewank(lst)
        elif self.name == "levy":
            return self.levy(lst)
        elif self.name == "michalewicz":
            return self.michalewicz(lst)
        elif self.name == "zakharov":
            return self.zakharov(lst)
        elif self.name == "ackley":
            return self.ackley(lst)

    def sphere(self, lst: list[float]) -> float:
        result = 0.0

        for item in lst:
            result += item ** 2

        return result

    def ackley(self, lst: list[float], a: float = 20, b: float = 0.2, c: float = 2 * pi) -> float:
        d = len(lst)
        sum1 = 0.0
        sum2 = 0.0

        for item in lst:
            sum1 += item ** 2
            sum2 += cos(c * item)

        sum1 /= d
        sum2 /= d

        sum1 = -b * sqrt(sum1)

        return -a * exp(sum1) - exp(sum2) + a + exp(1)

    def rastrigin(self, lst: list[float]) -> float:
        result = 10.0 * len(lst)

        for item in lst:
            result += (item ** 2 - 10 * cos(2 * pi * item))

        return result

    def rosenbrock(self, lst: list[float]) -> float:
        result = 0.0
        d = len(lst)
        lstMx = lst[0:(d - 1)]
        lstNxt = lst[1:d]

        for mx, nxt in zip(lstMx, lstNxt):
            result += (100 * (nxt - mx ** 2) ** 2 + (mx - 1) ** 2)

        return result

    def new_rosenbrock(self, lst: list[float]) -> float:
        result = 0.0
        d = len(lst)

        # from 1 to d-1
        for i in range(d-1):
            result += (100 * (lst[i+1] - lst[i]**2)**2 + (lst[i] - 1) ** 2)

        return result

    def griewank(self, lst: list[float]) -> float:
        sm = 0.0
        dot = 1.0

        for i, item in enumerate(lst):
            sm += item ** 2 / 4000
            dot *= cos(item / sqrt(i + 1))

        return sm - dot + 1

    def schwefel(self, lst: list[float]) -> float:
        d = len(lst)
        sm = 0.0

        for item in lst:
            sm += item * sin(sqrt(abs(item)))

        return 418.9829 * d - sm

    def levy(self, lst: list[float]) -> float:
        d = len(lst)
        w = []

        for item in lst:
            w.append(1.0 + (item - 1) / 4)

        result = sin(pi * w[0]) ** 2

        # d-2 because in math notation
        for i in range(d-2):
            result += ((w[i] - 1) ** 2) * (1 + 10 * sin(pi * w[i] + 1) ** 2)

        result += ((w[-1] - 1) ** 2) * (1 + sin(2 * pi * w[-1]) ** 2)

        return result

    def michalewicz(self, lst: list[float], m: int = 10) -> float:
        result = 0.0

        for i, item in enumerate(lst):
            result += sin(item) * sin(((i + 1) * (item ** 2)) / pi) ** (2 * m)

        return -1 * result

    def zakharov(self, lst: list[float]) -> float:
        sum1 = 0.0
        sum2 = 0.0

        for i, item in enumerate(lst):
            sum1 += item ** 2
            sum2 += 0.5 * (i + 1) * item

        return sum1 + sum2 ** 2 + sum2 ** 4

    # Validacni test
    def test(self) -> None:
        vals = [-1.25, 10.5, 22.19, 1.59753, -2.468, 0.1379]

        print(self.sphere(vals))
        print(self.schwefel(vals))
        print(self.rosenbrock(vals))
        print(self.rastrigin(vals))
        print(self.griewank(vals))
        print(self.levy(vals))
        print(self.michalewicz(vals))
        print(self.zakharov(vals))
        print(self.ackley(vals))
