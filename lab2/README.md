### Lab 2 solution 


This is an attempt for submission to lab2 using a generic algorithm

It's based on the following approach:

- Generate an initial population of individuals randomly.
- Evaluate the fitness of each individual in the population.
- Repeat as often as you like:
    - a) Select individuals with a good fitness score for reproduction.
    - b) Let them produce offspring.
    - c) Mutate these offspring.
    - d) Evaluate the fitness of each individual in the population.
    - f) Let the individuals with a bad fitness score die.

- Pick the individual with the highest fitness as the solution.

The alogorithm is not performing well at all at the moment and I will probably not have the time to improve

Some discussions about solutions and tweak of variables has been made with

- Karl Wennerström
- Angelica Ferlin
- Mathias Schmeckel
- Leonora Gomes

### Results

Calculated from 10 runs with each N with the following settings:

- NR_OF_GENERATIONS = 1000
- POPULATION_SIZE = 50
- OFFSPRING_SIZE = 20


| N    | Best | Worst |
|------|:----:|------:|
| 5    |  5   |     7 |
| 10   |  12  |    16 |
| 50   | 115  |   284 |
| 500  |      |       |
| 1000 |      |       |