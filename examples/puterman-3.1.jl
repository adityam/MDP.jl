using MDP

P = Matrix[ [ 0.5  0.5 
              0.0  1.0 ],
            [ 0.0  1.0
              0.5  0.5 ] ]

r = [ 5  10
     -1  -Inf ]

model = ProbModel(r, P; objective=:Max)

(v,g) = valueIteration(model; discount=0.95)

println(g)
println(v)
