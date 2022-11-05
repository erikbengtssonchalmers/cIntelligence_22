### Lab 2 solution 


This is an attempt for submission to lab2 using a generic algorithm. Due to lack
of time solution needs more work to function properly... TODO next week.

It's based on the following approach from the following source and materials:

- Generate an initial population of individuals randomly.
- Evaluate the fitness of each individual in the population.
- Repeat as often as you like:
    - a) Select individuals with a good fitness score for reproduction.
    - b) Let them produce offspring.
    - c) Mutate these offspring.
    - d) Evaluate the fitness of each individual in the population.
    - f) Let the individuals with a bad fitness score die.

- Pick the individual with the highest fitness as the solution.

The algorithm is not performing well at all at the moment and I will probably not have the time to improve
the current solution. However, by experimenting with the variables it would probably be 

Some discussions about possible approaches and solutions and tweak of variables has been made with

- Karl Wennerström
- Angelica Ferlin
- Mathias Schmeckel
- Leonora Gomes

### Results

Calculated from 10 runs with each N with the following settings:

- NR_OF_GENERATIONS = 1000
- POPULATION_SIZE = 50
- OFFSPRING_SIZE = 20


| N    | Best |  Worst |
|------|:----:|-------:|
| 5    |  5   |      7 |
| 10   |  12  |     16 |
| 50   | 115  |    284 |
| 500  | 1610 |   9016 |
| 1000 | 3948 | 22 441 |