using MDP
using Base.Test

P_01 = Matrix[ [ 0.5  0.5 
                 0.0  1.0 ],
               [ 0.0  1.0
                 0.5  0.5 ] ]

r_01 = [ 1 2 ]

@test_throws ErrorException ProbModel(r_01, P_01; objective = :Max)

P_02 = Matrix[ [ 0.5  0.4999 
                 0.0  1.0 ],
               [ 0.0  1.0
                 0.5  0.5 ] ]

r_02 = [ 1 2 ; 3 4]

@test_throws ErrorException ProbModel(r_02, P_02; objective = :Max)
