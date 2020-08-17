import math
import random
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from numba import njit


def get_ratio(v1, v2):
    denom = v1
    numer = v2
    if v1 < v2:
        denom = v2
        numer = v1
    ratio = numer / denom
    return ratio


@njit(fastmath=True)
def weighted_sum(W, X):
    """
    Computer w1*x1 + w2*x2 ..
    """
    return X @ W


@njit(parallel=True, fastmath=True)
def prepare_normalized_weighted_sum(W, X1, X2):
    weight_size = len(W)
    weight_male = W[: int(weight_size / 2)]
    weight_female = W[int(weight_size / 2) :]

    weighted_sum_m = weighted_sum(weight_male, X1)
    weighted_sum_f = weighted_sum(weight_female, X2)

    # normalize by weightsum
    w_male = np.sum(weight_male)
    w_female = np.sum(weight_female)
    weighted_sum_m_norm = np.divide(weighted_sum_m, w_male)
    weighted_sum_f_norm = np.divide(weighted_sum_f, w_female)

    return weighted_sum_m_norm, weighted_sum_f_norm


def accuracy(pred, actual_labels):
    predictions = [1 if w >= 0.5 else 0 for w in pred]
    accuracy1 = np.sum(
        (np.array(predictions) == np.array(actual_labels)) / len(actual_labels)
    )
    return accuracy1


def error(pred, actual_labels):
    accuracy1 = accuracy(pred, actual_labels)
    return 1 - accuracy1


def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (
                (values1[p] > values1[q] and values2[p] > values2[q])
                or (values1[p] >= values1[q] and values2[p] > values2[q])
                or (values1[p] > values1[q] and values2[p] >= values2[q])
            ):
                if q not in S[p]:
                    S[p].append(q)
            elif (
                (values1[q] > values1[p] and values2[q] > values2[p])
                or (values1[q] >= values1[p] and values2[q] > values2[p])
                or (values1[q] > values1[p] and values2[q] >= values2[p])
            ):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (
            values1[sorted1[k + 1]] - values2[sorted1[k - 1]]
        ) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (
            values1[sorted2[k + 1]] - values2[sorted2[k - 1]]
        ) / (max(values2) - min(values2))
    return distance


def crossover(a, b):
    """
    instead of negate : cross/exchange the w values
    only add . not to mutation
    """
    # #     print('a and b', a, a[0,0], a[0,1], type(b))
    r = random.random()
    if r > 0.5:
        return np.array(
            [mutation((a[0, 0] + b[0, 0]) / 2), mutation((a[0, 1] + b[0, 1]) / 2)]
        )
    else:
        return np.array(
            [mutation((a[0, 0] - b[0, 0]) / 2), mutation((a[0, 1] - b[0, 1]) / 2)]
        )


def randomly_mutate_population(population, mutation_probability):
    """
    Randomly mutate population with a given individual gene mutation
    probability. Individual gene may switch between 0/1.
    """
    # Apply random mutation
    random_mutation_array = np.random.random(size=(population.shape))

    random_mutation_boolean = random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = np.logical_not(
        population[random_mutation_boolean]
    )

    # Return mutation population
    return population


def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_w + (max_w - min_w) * random.random()
    #     print('solution is', solution)
    return solution


def breed_by_crossover(parent_1, parent_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length - 1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point], parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point], parent_1[crossover_point:]))

    # Return children
    return child_1, child_2


@njit(parallel=True, fastmath=True)
def mutate_population(population):
    """
    Randomly mutate population with a given individual gene mutation
    probability.
    """
    # Apply random mutation

    rows = population.shape[0]
    cols = population.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):

            mutation_prob = random.uniform(0, 1)
            item = population[x][y]
            if mutation_prob < 0.1:
                population[x][y] = item + random.uniform(0, 1)

    return population


def breed_by_crossover_inter_population(parent_1, parent_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    # Get length of chromosome
    chromosome_length = len(parent_1)

    child_1 = (parent_1 + parent_2) / 2

    child_2 = abs(parent_1 - parent_2) / 2
    return child_1, child_2
