using MDP

# This example describes a Markov decision process where
#
# S = {1, 2}
# A = {1, 2}

# r = S x A matrix

r = [ 5  10
     -1  -Inf ]


# Transition matrix
# P(1) = [0.5, 0.5; 0.0, 1.0]
# P(2) = [0.0, 1.0; 0.5, 0.5]
P = Matrix[ [ 0.5  0.5 
              0.0  1.0 ],
            [ 0.0  1.0
              0.5  0.5 ] ]

# Construct the MDP model
model = ProbModel(r, P; objective=:Max)

# Compute the optimal solution
(v,g) = valueIteration(model; discount=0.95, tolerance=1e-4)

println(g) # [1, 1]
println(v) # [-8.571401228528563,-19.999971289954992]

# For this model, the optimal solution is:

# g = [1, 1]
# w[1] = (5 - 5.5 β) / (1 -0.5 β)(1 - β)
# w[2] = -1/(1-β)

# which equals
# 
# w = [ -8.57142857142855, -19.999999999999982]

# As expected, v is within 1e-4 of w (in sup-norm). 

