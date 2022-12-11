#### Lab 3.1 and 3.2

The lab has been implemented by myself, but a lot of concepts and rulas has been discussed together with Karl Wennerström, also some discussions with Angelica Ferlin.

I've also been using code provided by the professor during the lectures, mainly for part 3.1.

# Description
The player taking the last object wins.

- Task 3.1: An agent using fixed rules based on nim-sum (i.e., an expert system)
- Task 3.2: An agent using evolved rules
- Task 3.3: An agent using minmax
- Task 3.4: An agent using reinforcement learning

### 3.1 - Expert system

Most of this solution has been provided from professor in class, so that it feels unnecessary
with to many comments. However, the solution uses the nim_sum and plays against a random opponent. As long as the nr 
of rows isn't too few (like 2 or 1), it basically always wins against a player playing random.

It's also possible for it to play other opponents but the hard coded game loop might have to be changed a bit.


### 3.2 Evolved Rules
The basic idea of the solution was to come up with some rules that several agents used in certain situations. These rules all
have a parameter telling us how many tiles to leave on a row in the case of this being applied. The parameters are randomized in the 
creation of the population and the idea is that the genes using the best tweaked parameters will win the most games and therefor become
our best individuals.

As an example, one of the rules is used when the game only consists of on active row. In this case, we obviuosly want to take all tiles on the last row 
and win the game which means that the ideal parameter for that rule is zero.

# Implementation 
The solution uses a (lamdba, mu)-strategy.

The current implementation is to use the provided cooked interface to find the cases on when to apply what rules. The agents playing is instances of 
Gene class and holds information about themselves, but also has a Strategy for when making a move. The strategies are a strategy pattern which makes it possible
to move with different concrete implementations.

We start by creating a population with N number of Genes and then let them face a random gene 10 different times. After each game the genes stats are updated.
Then two parents are selected using a tournament selection for crossover. Crossover in this case is basically just take parameter 1 from parent 1 and 2 from parent 2.

Mutation happens with prob P and it selects a random individual from the population and randomizes its parameter. 

The script currently runs quite slow and in the end it prints the individual with the highest fitness, the gene that has the highest win precentage after
10 generations. It has been showed when testing the solution that there might be a problem using only 2 rules and a random strategy. That leads that both players
starts with random movement and then only in the end practice their rules. So to improve, more rules should be implemented which is possible with the current deisgn.
Should also be less random which opponent to face. A gene could be able to try its luck against bad strategies, opotimal and OK ones.


### 3.3 An agent using minmax

Currently, the minmax agent can win against the optimal strategy if it can play first with 4 rows. Since the complexity of the algorithm is high, I've only managed to play
using max 5 rows in the game if mmy agent doesn't start.

### 3.4 An agent using reinforcement learning
