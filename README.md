# MDP

[![Build Status](https://travis-ci.org/adityam/MDP.jl.svg?branch=master)](https://travis-ci.org/adityam/MDP.jl)

The MDP package implements algorithms for Markov decision processes (MDP).

# Installation

MDP can be installed through Julia package manager 

    julia> Pkg.clone("https://github.com/adityam/MDP.jl")

# Supported algorithms

The MDP package currently implements the following algorithms

* **Infinite horizon discounted setup**

    - Value iteration 

Other algorithms related to MDPs will be implemented as time permits. Pull
requests welcome.

# Specifying a model

There are two ways to specify the MDP model:

1. By specifying the controlled transition matrix and the cost and reward
   function. 

2. By specifying the bellman update operator.

Perhaps the easiest way to understand these is by means of examples. 

## Specifying a model: probabilistic method

_This example is taken from Puterman, "Markov Decision Processes", Wiley 2005
(Chapter 3.1)_

Consider a Markov decision process where

- **States**  S = {1, 2}
- **Actions** A = {1, 2}
- **Rewards** r(s,a) given by 

        r(1,1) = 5      r(1,2) = 10
        r(2,1) = -1     r(2,2) = -Inf

    Note that `r(2,2) = -Inf` means that action 2 is infeasible at state 2.

- **Transition probabilities** P(s_next | s_current, action)

        P(1 | 1, 1) = 0.5       P(2 | 1, 1) = 0.5
        P(1 | 2, 1) = 0.0       P(2 | 2, 1) = 1.0

        P(1 | 1, 2) = 0.0       P(2 | 1, 2) = 1.0
        P(1 | 2, 2) = 0.5       P(2 | 2, 2) = 0.5

This model is specified as follows:

- Specify the rewards as a `S x A` matrix.

        r = [ 5  10
             -1  -Inf]

- Specify the transition matrix as a list `{ P( . | ., 1), P( . | ., 2) }`.


        P = { [ 0.5  0.5 
                0.0  1.0 ],
              [ 0.0  1.0
                0.5  0.5 ] }

- Specify an MDP model

        model = ProbModel(r, P; objective=:Max)

    `objective=:Max` specifies that we want to maximize reward. To minimize a
    cost function, use `objective=:Min`.

To solve this model using value iteration use

    (v,g) = valueIteration(model; discount=0.95)

`v` is the value function (stored as a `S` dimensional vector) nad `g` is the
policy (again, stored as a `S` dimensional vector). 

The optimal arguments for `valueIteration` are

* `discount` (default `0.95`): The discount factor.

* `iterations` (default `1_000`): The maximum number of iteration to be
  perfermoed.

* `tolerance` (default `1e-4`): Stop the iteration when sup-norm of `v - v_opt`
  is less than `tolerance`, where `v_opt` is the optimal solution.
    
* `intial_v` (default, an all zeros-vector). 

Here is the full example:

    using MDP

    P = { [ 0.5  0.5 
            0.0  1.0 ],
          [ 0.0  1.0
            0.5  0.5 ] }

    r = [ 5  10
         -1  -Inf ]

    model = ProbModel(r, P; objective=:Max)

    (v,g) = valueIteration(model; discount=0.95)

    print(v)


which outputs


    INFO: value iteration will converge in at most 20 iterations
    INFO: Reached precision 5.742009e-05 at iteration 18
    [-8.571401228528563,-19.999971289954992]
      
