import random
from itertools import accumulate
from operator import xor
from copy import deepcopy
from collections import namedtuple
import numpy as np

# Some global variables used to control population and game
POPULATION_SIZE = 100
NR_OF_ROWS = 11
OFFSPRING_SIZE = 50
MUTATION_PROB = 0.3
NR_OF_GENERATIONS = 10

Nimply = namedtuple("Nimply", "row, num_objects")


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


class Agent:
    def __init__(self, strategy):
        self.strategy = strategy
        self.games_played = 0
        self.games_won = 0
        self.score = 0

    def update_score(self, win):
        if win:
            self.games_won += 1
        self.games_played += 1
        self.score = self.games_won / self.games_played

    def execute(self, move):
        """
            Call depending on dynamic "configuration", we execute move using a strategy. Default random.
        """
        return self.strategy.move(move, None)


class Gene:
    """
        Gene class used in population. It holds information about the gene and how its performed. Also holds a strategy
        for making a move and a setter for that strategy
    """

    def __init__(self):
        self.parameters = self.populate()
        self.strategy = RandRule()
        self.games_played = 0
        self.games_won = 0
        self.score = 0

    def update_score(self, win):
        if win:
            self.games_won += 1
        self.games_played += 1
        self.score = self.games_won / self.games_played

    def populate(self):
        """
            key : strategy name
            value : how many objects to leave on each row
        """
        parameters = {'one_row_left': random.randint(1, NR_OF_ROWS),
                      'two_rows_left': random.randint(1, NR_OF_ROWS)}
        return parameters

    def set_strategy(self, new_strategy):
        """
            setter for strategy method
            :param new_strategy: Callable, strategy to execute when calling execute()
        """
        self.strategy = new_strategy

    def execute(self, move):
        """
            Call depending on dynamic "configuration", we execute move using a strategy. Default random.
        """
        return self.strategy.move(move, self.parameters)


class Strategy:
    """
        Abstract strategy class, defines move interface for specific implementations
    """

    def __init__(self):
        pass

    """
        Abstract method to be implemented in subclasses
    """

    def move(self, move, parameters):
        pass


class QLearning(Strategy):
    def __init__(self):
        super().__init__()

    def move(self, move, parameters):
        return None


class OneRowLeft(Strategy):
    """
        Strategy to leave x tiles in game when only one row is left
    """

    def __init__(self):
        super().__init__()

    def move(self, possible_moves, parameters):
        print('moving with rule 1')
        max_move = max(possible_moves, key=lambda i: i[1])
        units = max(max_move[1] - parameters['one_row_left'], 1)
        return Nimply(max_move[0], units)


class RandRule(Strategy):
    """
        Pure random strategy, used as default for all agents
    """

    def __init__(self):
        super().__init__()

    def move(self, possible_moves, parameters):
        print('im moving with a random strategy')
        move = possible_moves[random.randint(0, len(possible_moves) - 1)]
        return Nimply(move[0], move[1])


class TwoRowsLeft(Strategy):
    """
        Apply when there are two rows left. Then take x amount of elements from the row with most tiles left
    """

    def __init__(self):
        super().__init__()

    def move(self, possible_moves, parameters):
        # leave x amount of elements from the row with most tiles left
        print('Using two row strategy')
        max_move = max(possible_moves, key=lambda i: i[1])
        nr_of_tiles = max(max_move[1] - parameters['two_rows_left'], 1)
        return Nimply(max_move[0], nr_of_tiles)


class NimSumStrategy(Strategy):
    """
        Strategy to use nim sum strategy in game. Only used by nim sum agent. Logic provided in class
    """

    def __init__(self):
        super().__init__()

    def move(self, possible_moves, parameters):
        # leave x amount of elements from the row with most tiles left
        print('Moving w nimsum strategy')
        return next((bf for bf in possible_moves if bf[1] == 0), random.choice(possible_moves))[0]


class BestMoveStrategy(Strategy):
    """
        Strategy using minmax to minimize the opponents maximal win.
    """

    def __init__(self):
        super().__init__()

    def move(self, state, parameters):
        data = cook_status(state)
        print('moving with BestMoveStrategy')
        for ply in data['possible_moves']:
            tmp_state = deepcopy(state)
            tmp_state.nimming(ply)
            score = min_max(tmp_state, maximizing=False)
            if score > 0:
                break
        return ply


# Provided by professor, used for agent playing nim sum strategy
def nim_sum(state: Nim) -> int:
    *_, result = accumulate(state.rows, xor)
    return result


# Random strategy used in part 3.1
def pure_random(state: Nim) -> Nimply:
    move_dict = cook_status(state)
    row, nr = move_dict['possible_moves'][random.randint(0, len(move_dict['possible_moves']) - 1)]
    return Nimply(row, nr)


# Provided from professor with some small changes
def optimal_strategy(state: Nim) -> Nimply:
    data = cook_status(state)
    return next((bf for bf in data["brute_force"] if bf[1] == 0), random.choice(data["brute_force"]))[0]


# Provided from professor with some small changes
def cook_status(state: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    if cooked['active_rows_number'] == 1:
        cooked['last_row'] = cooked["possible_moves"][0][0]

    cooked["shortest_row"] = min((x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1])[0]
    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[0]
    cooked["nim_sum"] = nim_sum(state)

    brute_force = list()
    for m in cooked["possible_moves"]:
        tmp = deepcopy(state)
        tmp.nimming(m)
        brute_force.append((m, nim_sum(tmp)))
    cooked["brute_force"] = brute_force
    return cooked


"""
def play_nim(game_nim: Nim):
    #Initial game function
    strategy = [pure_random, optimal_strategy]
    player = 0
    print(f'initial state: {game_nim}')
    while game_nim:
        ply = strategy[player](game_nim)
        game_nim.nimming(ply)
        print(f'Board after player {player}\'s move: {game_nim}')
        player = 1 - player
    print(f'player {1 - player} won')
"""


def play_nim_game(game_nim, agent1, agent2):
    """
        Playing game using agents with different strategies
    """
    players = [agent1, agent2]
    index = 0
    print(f'initial state: {game_nim}')
    while game_nim:
        ply = evaluate_game(deepcopy(game_nim), players[index])
        game_nim.nimming(ply)
        print(f'Board after player: {index}\'s move: {game_nim}')
        index = 1 - index
    print(f'player {1 - index} won')
    winner = players[1 - index]
    loser = players[index]
    return winner, loser


def evaluate_game(state, gene):
    """
        get information from game and call strategies
    """
    game_dict = cook_status(state)

    if type(gene) is LearningAgent:
        move = gene.execute(state)
        return move

    # fix so that it's easy to face both random and nim sum strategies, needs refactor.
    if type(gene) is Agent:
        if type(gene.strategy) is NimSumStrategy:
            return gene.execute(game_dict['brute_force'])
        elif type(gene.strategy) is BestMoveStrategy:
            return gene.execute(state)
        else:
            return gene.execute(game_dict['possible_moves'])

    if game_dict['active_rows_number'] == 1:
        gene.set_strategy(OneRowLeft())
        return gene.execute(game_dict['possible_moves'])

    elif game_dict['active_rows_number'] == 2:
        gene.set_strategy(TwoRowsLeft())
        return gene.execute(game_dict['possible_moves'])

    else:
        gene.set_strategy(RandRule())
        ply = gene.execute(game_dict['possible_moves'])
        return ply


def tournament(p1, p2):
    if p1.score > p2.score:
        return p1
    else:
        return p2


def create_offspring(population):
    for i in range(OFFSPRING_SIZE):
        tmp = random.sample(population, 4)
        parent1 = tournament(tmp[0], tmp[2])
        parent2 = tournament(tmp[1], tmp[3])
        new_gene = crossover(parent1, parent2)
        population.append(new_gene)
    return population


def crossover(p1, p2):
    first_val = p1.parameters['one_row_left']
    second_val = p2.parameters['two_rows_left']
    child = Gene()
    child.parameters['one_row_left'] = first_val
    child.parameters['two_rows_left'] = second_val
    return child


def mutate(population):
    rand_index = random.randint(0, len(population) - 1)
    gene_to_mutate = population[rand_index]
    gene_to_mutate.parameters['one_row_left'] = random.randint(0, NR_OF_ROWS)
    gene_to_mutate.parameters['two_rows_left'] = random.randint(0, NR_OF_ROWS)
    population.append(gene_to_mutate)
    return population


def min_max(game: Nim, maximizing: bool):
    if not game:
        return -1 if maximizing else 1

    status = cook_status(game)
    possible_moves = status['possible_moves']

    new_states = []
    for move in possible_moves:
        tmp_state = deepcopy(game)
        tmp_state.nimming(move)
        new_states.append(tmp_state)

    if maximizing:
        scores = [
            min_max(new_state, not maximizing)
            for new_state in new_states
        ]
        return max(scores)
    else:
        # do optimal strategy to reduce complexity
        new_state = deepcopy(game)
        ply = optimal_strategy(new_state)
        new_state.nimming(ply)
        scores = [
            min_max(new_state, not maximizing)
        ]
        return min(scores)


def play_game_with_expert(size, first_agent, second_agent):
    """
    Lab 3.1
    """
    game = Nim(size)
    play_nim_game(game, first_agent, second_agent)


def evolutionary_strategy():
    """
    Lab 3.2
    """
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(Gene())

    for _ in range(NR_OF_GENERATIONS):
        # play games with genes vs genes (not facing each other) and save score.
        for gene in population:
            k = 0
            while k < 10:
                rand_opponent = population[random.randint(0, len(population) - 1)]
                if rand_opponent is not gene:
                    game = Nim(NR_OF_ROWS)
                    winner, loser = play_nim_game(game, gene, rand_opponent)
                    winner.update_score(True)
                    loser.update_score(False)
                    k += 1

        population = create_offspring(population)
        if random.random() < MUTATION_PROB:
            population = mutate(population)
        population = sorted(population, key=lambda x: x.score, reverse=True)[:POPULATION_SIZE]

    print(f'best gene: {population[0].parameters}, score {population[0].score}')


def play_game_with_minmax():
    play_nim_game(Nim(4), agent1=Agent(BestMoveStrategy()), agent2=Agent(NimSumStrategy()))


class LearningAgent:
    """
    TO BE USED FOR REINFORCEMENT LEARNING (Q learning).

    """

    def __init__(self, exploration_treshold, learning_rate, disc_rate):
        self.q = {}  # holds a state and its values
        self.strategy = QLearning()
        self.previous_move = None
        self.previous_state = None
        self.exploration_treshold = exploration_treshold
        self.learning_rate = learning_rate
        self.discount_rate = disc_rate
        self.PENALTY = -1
        self.REWARD = 1

    def reset_moves(self):
        self.previous_move = None
        self.previous_move = None

    def add_moves(self, state, possible_moves):
        for move in possible_moves:
            if state.rows not in self.q.keys():
                self.q[state.rows] = {}
            if move not in self.q[state.rows].keys():
                self.q[state.rows][move] = np.random.uniform(0.0, 0.01)

    def best_move(self, state, possible_moves):
        curr_max = 0
        best_move = None
        for m in self.q[state.rows].keys():
            if self.q[state.rows][m] > curr_max and m in possible_moves:
                best_move = m
                curr_max = self.q[state.rows][m]
        if not best_move:
            return possible_moves[0]
        return best_move

    def policy(self, state, possible_moves):
        rand = np.random.random()
        if rand >= self.exploration_treshold:
            if len(possible_moves) == 1:
                return possible_moves[0]
            else:
                return self.best_move(state, possible_moves)
        else:
            rand = random.sample(possible_moves, 1)[0]
            return rand

    def update_q(self, state, data):
        # game is not over
        move = self.policy(state, data)

        if self.previous_move:
            # it's not the first move of the game
            next_state = deepcopy(state)
            next_state.nimming(Nimply(move[0], move[1]))
            reward = 1 if next_state is None else 0

            max_val = max([self.q[state.rows][move] for move in data])

            # update q value according to algorithm
            self.q[self.previous_state.rows][self.previous_move] += self.learning_rate * (
                        (reward + self.discount_rate) * (
                            max_val - self.q[self.previous_state.rows][self.previous_move]))
        self.previous_state = state
        self.previous_move = move
        return move

    def react_on_loss(self):
        self.q[self.previous_state.rows][self.previous_move] += self.learning_rate * (
                    self.PENALTY - self.q[self.previous_state.rows][self.previous_move])

    def lower_learning_rate(self):
        if self.learning_rate - 0.0005 >= 0.1:
            self.learning_rate -= 0.0005
        else:
            self.learning_rate = 0.1


    def execute(self, state):
        data = cook_status(state)
        self.add_moves(state, data['possible_moves'])
        move = self.update_q(state, data['possible_moves'])
        return Nimply(move[0], move[1])


def play_with_q_learning():
    nr_of_games = 1000
    q_agent = LearningAgent(exploration_treshold=0.1, learning_rate=0.4, disc_rate=0.6)
    opponents = [Agent(RandRule()), Gene(), Agent(NimSumStrategy())]
    wins = 0
    played = 0
    indx = 0
    for _ in range(100_000):
        opponent = opponents[indx]
        winner, loser = play_nim_game(Nim(3), agent1=q_agent, agent2=opponent)
        if type(winner) is LearningAgent:
            wins += 1
        else:
            q_agent.react_on_loss()
        played += 1
        q_agent.reset_moves()
        print(opponent.__dict__)
        if played == 100 or played == 35000:
            indx += 1
    print(wins / played)
    print(q_agent.__dict__)


def main():
    # uncomment this to run part 3.1
    # play_game_with_expert(size= 5, first_agent= Agent(NimSumStrategy()), second_agent= Agent(RandRule()))

    # uncomment below to run part 3.2
    # evolutionary_strategy()

    # uncomment to run part 3.3
    # play_game_with_minmax()

    # part 3.4
    # play_with_q_learning()
    pass

if __name__ == "__main__":
    main()
