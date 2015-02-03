using MDP
using Base.Test

# Puterman, Example 3.1

P_01 = Matrix[ [ 0.5  0.5 
                 0.0  1.0 ],
               [ 0.0  1.0
                 0.5  0.5 ] ]

r_01 = [ 5  10
        -1  -Inf ]

model_01 = ProbModel(r_01, P_01; objective = :Max)

discount_01  = 0.95
tolerance_01 = 1e-3

(v_01, g_01) = valueIteration(model_01; discount=discount_01, tolerance=tolerance_01)

v_opt_01 = [ (5.0 - 5.5 * discount_01)/( (1.0 - 0.5 * discount_01) * (1.0 - discount_01) ), 
            -1.0 / ( 1 - discount_01 ) ]

@test_approx_eq_eps v_01 v_opt_01 tolerance_01

# Test for sparse matrices
P_02 = SparseMatrixCSC[sparse([0.5 0.5; 0.0 1.0]),
                       sparse([0.0 1.0; 0.5 0.5]) ]

r_02 = [ 5  10
        -1  -Inf ]

model_02 = ProbModel(r_02, P_02; objective = :Max)

discount_02  = 0.95
tolerance_02 = 1e-3

(v_02, g_02) = valueIteration(model_02; discount=discount_02, tolerance=tolerance_02)

@test_approx_eq_eps v_02 v_opt_01 tolerance_02
