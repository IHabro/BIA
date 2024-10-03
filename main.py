import pandas as pd
from numpy import pi
from solution import Solution
from salesman import Salesman
from Ants import Colony

# Main
if __name__ == "__main__":
    bd = [[-5.12, 5.12], [-500, 500], [-5, 10], [-5.12, 5.12], [-600, 600], [-10, 10], [0, pi], [-5, 10], [-32.768, 32.768]]
    fs = ["sphere", "schwefel", "rosenbrock", "rastrigin", "griewank", "levy", "michalewicz", "zakharov", "ackley"]
    excelData = []

    if len(bd) == len(fs):
        for f, b in zip(fs, bd):
            sol = Solution(2, b, f)
            excelData.append(sol.comparison())
            print("{} done".format(f))

    with pd.ExcelWriter("HAB0065_HW10_data_up.xlsx") as writer:
        for i in range(len(excelData)):
            df = pd.DataFrame(excelData[i], columns=["DE", "PSO", "SOMA", "FA", "TLBO"])
            df.to_excel(writer, sheet_name=fs[i])

    # cities = ["Prague", "Bratislava", "Berlin", "Budapest", "Moscow", "Ankara", "Sisburg", "Aufgang", "Normandy", "Warsaw"]
    # tsm = Salesman(cities, [100, 100])
    # tsm.genetic_algorithm(20, 10000, len(cities), 0.5)

    # colony = Colony([5, 20, len(cities)], [1, 2, 0.5], [100, 100], cities)
    # colony.anting()
