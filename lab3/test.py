import logging
import random
from itertools import accumulate
from operator import xor
from collections import namedtuple
from copy import deepcopy

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


def nim_sum(state: Nim) -> int:
    *_, result = accumulate(state.rows, xor)
    return result


def pure_random(state: Nim) -> Nimply:
    move_dict = cook_status(state)
    row, nr = move_dict['possible_moves'][random.randint(0, len(move_dict['possible_moves']) - 1)]
    return Nimply(row, nr)


def optimal_strategy(state: Nim) -> Nimply:
    data = cook_status(state)
    return next((bf for bf in data["brute_force"] if bf[1] == 0), random.choice(data["brute_force"]))[0]


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


def play_nim(game_nim: Nim):
    strategy = [pure_random, optimal_strategy]
    player = 0
    print(f'initial state: {game_nim}')
    while game_nim:
        ply = strategy[player](game_nim)
        game_nim.nimming(ply)
        print(f'Board after player {player}\'s move: {game_nim}')
        player = 1 - player
    print(f'player {1 - player} won')


# game = Nim(11)
# play_nim(game)


# LAB 3.2 BELOW

# Some global variables used to control population and game
POPULATION_SIZE = 100
NR_OF_ROWS = 11


def play_nim_game(game_nim, agent):
    players = [agent[0], agent[1]]
    index = 0
    print(f'initial state: {game_nim}')
    while game_nim:
        ply = evaluate_game(game_nim, players[index])
        game_nim.nimming(ply)
        print(f'Board after player {players[index]}\'s move: {game_nim}')
        index = 1 - index
    print(f'player {1 - index} won')


def evaluate_game(state, gene):
    # get information from game
    game_dict = cook_status(state)

    # evaluate information based on most important rules first, return Nimply from gene class

    # TODO: FIX TO NOT TAKE TO MANY BRICKS
    if game_dict['active_rows_number'] == 1:
        move = game_dict['last_row']
        gene.set_strategy(OneRowLeft())
        ply = gene.execute(move)
        return ply
    #elif game_dict['active_rows_number'] == 2:
    #    pass
    else:
        move = game_dict['possible_moves'][random.randint(0, len(game_dict['possible_moves']) - 1)]
        gene.set_strategy(RandRule())
        ply = gene.execute(move)
        return ply


class Gene:
    def __init__(self):
        self.parameters = self.populate()
        self.strategy = RandRule()

    def populate(self):
        """
            key : strategy name
            value : how many objects to leave on each row
        """
        parameters = {'one_row_left': random.randint(1, NR_OF_ROWS),
                      'one_row_to_win': random.randint(1, NR_OF_ROWS)}
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


class OneRowLeft(Strategy):
    """
        Strategy to use to win the game when the game only consists x units on one single row
    """

    def __init__(self):
        super().__init__()

    def move(self, move, parameters):
        print('movin with rule 1')
        # make implementation and return Nimply
        ply = Nimply(move, parameters['one_row_left'])
        return ply


class RandRule(Strategy):
    """
        Strategy to use to win the game when the game only consists x units on one single row
    """

    def __init__(self):
        super().__init__()

    def move(self, move, parameters):
        print('im moving with a random strategy')
        ply = Nimply(move[0], move[1])
        return ply


game = Nim(NR_OF_ROWS)
gene1 = Gene()
gene2 = Gene()

play_nim_game(game, [gene1, gene2])
