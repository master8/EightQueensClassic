import numpy as np
import random


class Solver_8_queens:
    __pop_size = 10
    __cross_prob = 1.0
    __mut_prob = 1.0

    __board_size = 8

    __code_size = int(np.log2(__board_size))
    __addition_size = 8 - __code_size
    __full_code_size = int(__code_size * __board_size)

    def __init__(self, pop_size=30, cross_prob=1.0, mut_prob=1.0):
        self.__pop_size = pop_size
        self.__cross_prob = cross_prob
        self.__mut_prob = mut_prob

    def solve(self, min_fitness=None, max_epochs=None):
        best_fit = 0.0
        epoch_num = 0

        population = self.__generate_population()

        while best_fit != 1.0:
            if min_fitness is not None and best_fit >= min_fitness:
                break

            if max_epochs is not None and epoch_num >= max_epochs:
                break

            fit_values = self.__count_fit_values(population)
            parents = self.__reproduction(population, fit_values)
            children = self.__crossover(parents)
            self.__mutate(children)
            population, best_fit = self.__select(population, children, fit_values)

            epoch_num += 1

        visualization = self.__generate_visualization(population[len(population) - 1])

        return best_fit, epoch_num, visualization

    def __generate_population(self):
        initial_population = np.zeros((self.__pop_size, self.__full_code_size), dtype=np.ubyte)

        for chromosome in initial_population:
            for i in range(0, len(chromosome)):
                chromosome[i] = random.randrange(0, 2)

        return initial_population

    def __count_fit_values(self, population):
        fit_values = np.zeros(len(population))
        for i, chromosome in enumerate(population):
            fit_values[i] = (self.__fit_fun(self.__chromosome_into_int_code(chromosome)))

        return fit_values

    def __reproduction(self, population, fit_values):
        wheel = []

        for index, value in enumerate(fit_values):
            size = int(round(value / np.sum(fit_values) * 100))
            for _ in range(0, size):
                wheel.append(index)

        parents = np.zeros((len(population), self.__full_code_size), dtype=np.ubyte)

        for i in range(0, len(population)):
            parents[i] = population[wheel[random.randrange(0, len(wheel))]]

        return parents

    def __crossover(self, parents):
        children = np.zeros((len(parents), self.__full_code_size), dtype=np.ubyte)

        for x in range(0, int(self.__pop_size / 2)):
            if random.randrange(0, 100) < self.__cross_prob * 100:
                y = int(self.__pop_size / 2) + x
                first_parent = parents[x]
                second_parent = parents[y]

                point = random.randrange(1, self.__full_code_size)

                children[x] = np.concatenate((first_parent[:point], second_parent[point:]))
                children[y] = np.concatenate((second_parent[:point], first_parent[point:]))

        return children

    def __mutate(self, children):
        for chromosome in children:
            if random.randrange(0, 100) < self.__mut_prob * 100:
                bit = random.randrange(0, len(chromosome))
                if chromosome[bit] == 1:
                    chromosome[bit] = 0
                else:
                    chromosome[bit] = 1

    def __select(self, population, children, fit_values):
        children_fit_values = self.__count_fit_values(children)
        full_population = np.concatenate((population, children))
        full_fit_values = np.concatenate((fit_values, children_fit_values))

        new_population = full_population[full_fit_values.argsort()][self.__pop_size:]

        return new_population, np.max(full_fit_values)

    def __generate_visualization(self, solution):
        result = ''
        for gen in self.__chromosome_into_int_code(solution):
            result += '+' * gen + 'Q' + '+' * (self.__board_size - gen - 1) + '\n'

        return result

    def __fit_fun(self, chromosome_int_code):
        count = len(chromosome_int_code) - len(np.unique(chromosome_int_code))

        for x1, y1 in enumerate(chromosome_int_code):
            for x2 in range(x1 + 1, len(chromosome_int_code)):
                if self.__is_diagonal_conflict(x1, y1, x2, chromosome_int_code[x2]):
                    count += 1

        return 1 / (count + 1)

    @staticmethod
    def __is_diagonal_conflict(x1, y1, x2, y2):
        if x1 == x2 or y1 == y2:
            return False
        diff_x = int(x1) - int(x2)
        diff_y = int(y1) - int(y2)
        return abs(diff_x / diff_y) == 1

    def __chromosome_into_int_code(self, chromosome_bin_code):
        full_bin_code = self.__generate_full_code(chromosome_bin_code[:self.__code_size])

        for k in range(self.__code_size, len(chromosome_bin_code), self.__code_size):
            full_bin_code = np.concatenate((
                full_bin_code,
                self.__generate_full_code(chromosome_bin_code[k:][:self.__code_size])
            ))

        return np.packbits(full_bin_code)

    def __generate_full_code(self, bin_code):
        bin_addition = np.zeros((1, self.__addition_size), dtype=np.ubyte)[0]
        return np.concatenate((bin_addition, bin_code))
