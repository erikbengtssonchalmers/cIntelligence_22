### Lecture with Lab1 comment

- Local search
**Hill climbing**. Solve optimization problem, related **gradient ascent** but doesn't require gradient

First improvement - Hill climbing or Random mutation HC,

vs

Steepest-steep Hill climber (ascent) 


#### stopping condition
- Best sol found
- nr of evaluations / steps
- Wall clock time
- Steady state - give up if chance of improvement is low

#### Simulated annealing
Hill Climb w p !=0 of accepting a worsening solution s' where 

Metafor, guy drunk walking on line test... random in the beginning but stabilizing later


#### Tabu search
Idea is to do something, mark it as bad if it's bad and then don't do it again or remove the bad solution from space

#### iterated local search

Cleverer version of HC, using random restarts
Returning the best previous solution found so far usually is a good way


Live code example on hill climbing approach can be found online....

