using MDP

P = { [ 0.5  0.5 
        0.0  1.0 ],
      [ 0.0  1.0
        0.5  0.5 ] }

r = [ 5  10
     -1  -100 ]

model = ProbModel(r, P; objective=:Max)

(v,g) = valueIteration(model; discount=0.95, tolerance=0.01)

print(v)
