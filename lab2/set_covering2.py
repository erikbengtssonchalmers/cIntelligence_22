import random
import logging
from pprint import pprint
"""

- Generate an initial population of individuals randomly.
- Evaluate the fitness of each individual in the population.

- Repeat as often as you like:
    a) Select individuals with a good fitness score for reproduction.
    b) Let them produce offspring.
    c) Mutate these offspring.
    d) Evaluate the fitness of each individual in the population.
    f) Let the individuals with a bad fitness score die.

(We're finished...)
- Pick the individual with the highest fitness as the solution.

"""

N = 10
NR_OF_GENERATIONS = 100


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
    for i in range(random.randint(10, 20)):
        population.append(random.sample(full_input, random.randint(1, len(full_input))))
    return population


# check if one solution is valid
def goal_check(curr, final_solution):
    return curr == set(range(N))


def fitness_function(entry, goal_set):
    flat_entry = [item for sublist in entry[0] for item in sublist]
    duplicates = len(flat_entry) - len(set(flat_entry))
    missing_elements = len(set(flat_entry).difference(goal_set))
    return -(1 * missing_elements) + (-5 * duplicates)


# Calculate fitness function and update
def calculate_fitness(current_solution):
    goal_set = set(list(range(N)))

    flat_individual = [item for sublist in current_solution[0] for item in sublist]
    return fitness_function(flat_individual, goal_set)


def select_parents(population):
    fitness_sum = sum(x[1] for x in population)
    print(fitness_sum)


def crossover():
    pass

# mutate solution and see if it gets better... don't know how yet
def mutate_solution(curr_solution):
    return list()





def main():
    logging.basicConfig(level=logging.DEBUG)
    # solution_set = set(list(range(N)))
    problem_space = problem(N, seed=42)
    population = select_rand_solution(problem_space)

    # should hold current solution with the calculated fitness
    current_individuals = []

    # setup data structure, list of tuples containing ([entries], fitness)
    for individual in population:
        current_individuals.append((individual, fitness_function(individual, set(range(N)))))

    counter = 0
    while counter < NR_OF_GENERATIONS:
        for individual in current_individuals:
            pass
            #itness = calculate_fitness(individual)

            # fix later...
            #current_individuals.remove(individual)
            #current_individuals.append((individual[0], fitness))




        current_individuals = sorted(current_individuals, key=lambda i: i[1], reverse=True)
        counter += 1

    select_parents(current_individuals)
    pprint(current_individuals)


if __name__ == "__main__":
    main()
