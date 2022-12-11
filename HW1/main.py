import numpy as np
import random


def ackley(variables_values=[0.0, 0.0]):
    x1, x2 = variables_values
    func_value = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.exp(1) + 20
    return func_value


def fitness_function(variables_values=[0.0, 0.0]):
    x1, x2 = variables_values
    result = (4*x1**2)-(2.1*x1**4)+((x1**6)/3)+(x1*x2)-(4*x2**2)+(4*x2**4)
    return result


def target_function(variables_values=[0.0, 0.0]):
    return fitness_function(variables_values)


def initial_position(grasshoppers=5, min_values=[-5.0, -5.0], max_values=[5.0, 5.0], target_function=target_function):
    position = np.zeros((grasshoppers, len(min_values) + 1))
    for i in range(0, grasshoppers):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0:position.shape[1] - 1])
    return position


def s_function(r, F, L):
    s = F * np.exp(-r / L) - np.exp(-r)
    return s


def build_distance_matrix(position):
    a = position[:, :-1]
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    return np.sqrt(np.einsum('ijk,ijk->ij', b - a, b - a)).squeeze()


def update_position(position, best_position, min_values, max_values, C, F, L, target_function):
    sum_grass = 0
    distance_matrix = build_distance_matrix(position)
    distance_matrix = 2 * (distance_matrix - np.min(distance_matrix)) / (np.ptp(distance_matrix) + 0.00000001) + 1
    np.fill_diagonal(distance_matrix, 0)
    for i in range(0, position.shape[0]):
        for j in range(0, len(min_values)):
            for k in range(0, position.shape[0]):
                if (k != i):
                    sum_grass = sum_grass + C * ((max_values[j] - min_values[j]) / 2) * s_function(
                        distance_matrix[k, i], F, L) * ((position[k, j] - position[i, j]) / distance_matrix[k, i])
            position[i, j] = np.clip(C * sum_grass + best_position[0, j], min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0:position.shape[1] - 1])
    return position


# GOA Function
def goa_algorithm(grasshoppers=5, min_values=[-5.0, -5.0], max_values=[5.0, 5.0], c_min=0.00004, c_max=1,
                  iterations=1000, F=0.5, L=1.5, target_function=target_function, print_iter=True, init_position=None):
    iter = 0
    if init_position is None:
        position = initial_position(grasshoppers, min_values, max_values, target_function)
    else:
        position = init_position
    best_position = np.copy(position[np.argmin(position[:, -1]), :].reshape(1, -1))
    while iter <= iterations:
        if print_iter:
            print('Iteration = ', iter, ' f(x) = ', best_position[0, -1])
        C = c_max - iter * ((c_max - c_min) / iterations)
        position = update_position(position, best_position, min_values, max_values, C, F, L,
                                   target_function=target_function)
        if np.amin(position[:, -1]) < best_position[0, -1]:
            best_position = np.copy(position[np.argmin(position[:, -1]), :].reshape(1, -1))
        iter = iter + 1
    return best_position


if __name__ == '__main__':
    #print(initial_position(grasshoppers=4))
    #initial_positions = np.zeros((4, 3))
    init_array = [[3.1472, 1.3236, 166.9550], [4.0579, -4.0246, 1953.1], [-3.7301, -2.2150, 631.9194], [4.1338, 0.4688, 1119.6]]
    initial_positions = np.array(init_array)
    print(initial_positions)
    best_position = goa_algorithm(grasshoppers=4, iterations=9, init_position=initial_positions)
    #print(target_function([3.1472, 1.3236]))
    #print(target_function([-3.7301, -2.0246]))
    print(best_position)
    #print(best_position[0, 0:2])
    print(target_function(best_position[0, 0:2]))
