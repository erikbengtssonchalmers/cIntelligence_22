import random
import logging
from pprint import pprint


N = 50
NR_OF_GENERATIONS = 1000
POPULATION_SIZE = 50
OFFSPRING_SIZE = 20


# search space generator, supplied by assignment
def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]


# Should be used to init solution space, return a list of list
def select_rand_solution(full_input):
    population = []
    random.seed(None)
    for i in range(POPULATION_SIZE):
        population.append(random.sample(full_input, random.randint(1, len(full_input))))
    return population


# check if one solution is valid
def goal_check(curr):
    return set(curr) == set(range(N))


def fitness_function(entry, goal_set):
    duplicates = len(entry) - len(set(tuple(entry)))
    miss = len(goal_set.difference(set(entry)))
    return (-1000 * miss) - duplicates


def calculate_fitness(individual):
    flat_individual = [item for sublist in individual for item in sublist]
    fitness_val = fitness_function(flat_individual, set(range(N)))
    return fitness_val


def select_parents(population):
    nr_of_boxes = int(POPULATION_SIZE * (POPULATION_SIZE + 1) / 2)
    random.seed(None)
    random_wheel_nr = random.randint(1, nr_of_boxes)
    parent_number = POPULATION_SIZE
    increment = POPULATION_SIZE - 1
    curr_parent = 0
    while random_wheel_nr > parent_number:
        curr_parent += 1
        parent_number += increment
        increment -= 1
    return population[curr_parent]


# randomize an index and merge 0-index from parent 1 and index-len of parent two, mutate with 5% chance
def crossover(first_parent, second_parent):
    slice_index_one = random.randint(0, min(len(first_parent[0]) - 1, len(second_parent[0]) - 1))
    child = first_parent[0][:slice_index_one] + second_parent[0][slice_index_one:]
    return child


# mutate child and return
def mutate_child(individual, problem_space):
    index = random.randint(0, len(individual) - 1)
    random_list = problem_space[random.randint(0, len(problem_space) - 1)]
    random_gene = random_list[random.randint(0, len(random_list) - 1)]
    individual = individual[:index] + individual[index+1:] + [random_gene]
    return individual


def update_population(population, new_children):
    new_population = population + new_children
    sorted_population = sorted(new_population, key=lambda i: i[1], reverse=True)
    return sorted_population[:POPULATION_SIZE]


def main():
    logging.basicConfig(level=logging.DEBUG)
    problem_space = problem(N, seed=42)
    population = select_rand_solution(problem_space)

    # should hold current population with the calculated fitness
    current_individuals = []

    # setup data structure, list of tuples containing ([entries], fitness) and sort
    for individual in population:
        current_individuals.append((individual, calculate_fitness(individual)))

    current_individuals = sorted(current_individuals, key=lambda l: l[1], reverse=True)

    counter = 0
    while counter < NR_OF_GENERATIONS:
        # a) Select individuals with a good fitness score for reproduction.
        cross_over_list = []
        for i in range(OFFSPRING_SIZE):
            parent_one = select_parents(current_individuals)
            parent_two = select_parents(current_individuals)

            # b) Let them produce offspring. Mutate with 5% chance
            tmp_child = crossover(parent_one, parent_two)
            if random.random() > 0.95:
                tmp_child = mutate_child(tmp_child, population)

            cross_over_list.append((tmp_child, calculate_fitness(tmp_child)))

        current_individuals = update_population(current_individuals, cross_over_list)
        counter += 1

    logging.info(f'Best solution for N={N} was {current_individuals[0][0]} with a weight of {sum(len(_) for _ in current_individuals[0][0])}')


if __name__ == "__main__":
    main()
