using MDP
using Base.Test

P_01 = Matrix[ [ 0.5  0.5 
                 0.0  1.0 ],
               [ 0.0  1.0
                 0.5  0.5 ] ]

r_01 = [ 1 2 ; 3 4]

r_01 = [ 1 2 ]

@test_throws ErrorException model_01 = ProbModel(r_01, P_01; objective = :Max)

