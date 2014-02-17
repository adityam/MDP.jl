using MDP

# Only needed for rle function
using StatsBase

const L = 50
const M = 30

#alpha = [0.75 0.75]
alpha = [0.75 0.75]

const P = [ alpha[1]		1-alpha[1]
	         1 - alpha[2]  alpha[2]  ]

p1 = p2 = 0.3

const Q1 = [ 1 - p1   p1 
              0        1 ]

const Q2 = [ 1 - p2   p2 
              0        1 ]

const xi = [ P^n for n = 1:M+1] 

function probabilityVector(Q, n)
    if n < L+2
        [1 0]*Q^n
    else # L+2 component corresponds to z_inf
        [0;1]
    end
end

z1 = [ probabilityVector(Q1, n)  for n = 1:L+2] 
z2 = [ probabilityVector(Q2, n)  for n = 1:L+2] 

r = 1.0
c = 0.1

function bellmanUpdate(v; discount=1.0)
    # Assume that v is L+2 x L+2 x 2 x M+1
    v_next = zeros(size(v)) :: Array{Float64}
    g_next = zeros(size(v))

    # v_next(k, ell, s, m)
    # k   = last time state of user 1 was observed
    # ell = last time state of user 2 was observed
    #  s  = last observed state of channel {Idle, Busy}
    #  m  = last time the state of channel was observed
    for m = 1:M+1, s = 1:2, ell = 1:L+2, k = 1:L+2
        next_m = min(m+1, M+1)
        next_ell = (ell < L+1)? ell+1 : ell
        next_k = (k < L+1)? k+1 : k

        q00 = discount * v[next_k, next_ell, s, next_m]

        q10 = ( z1[k][2] * xi[m][s,1] * r - z1[k][2] * c
              + discount * ( z1[k][1] * v[1, next_ell, s, next_m]
                           + z1[k][2] * xi[m][s,1] * v [1, next_ell, 1, 1]
                           + z1[k][2] * xi[m][s,2] * v [L+2, next_ell, 2, 1]
                           )
              )

        q01 = ( z2[ell][2] * xi[m][s,1] * r - z2[ell][2] * c
              + discount * ( z2[ell][1] * v[next_k, 1, s, next_m]
                           + z2[ell][2] * xi[m][s,1] * v [next_k, 1, 1, 1]
                           + z2[ell][2] * xi[m][s,2] * v [next_k, L+2, 2, 1]
                           )
              )

        q11 = ( ( z1[k][2] * z2[ell][1] + z1[k][1] * z2[ell][2] ) * xi[m][s,1] * r
              - ( z1[k][2] + z2[ell][2] ) * c
              + discount * ( z1[k][1] * z2[ell][1] * v[1, 1, s, next_m]
                           + (z1[k][2] * z2[ell][1] + z1[k][1] * z2[ell][2]) * xi[m][s,1] * v[1, 1, 1, 1]
                           + z1[k][2] * z2[ell][2] * xi[m][s,1] * v[L+2, L+2, 1, 1]
                           + (z1[k][2] + z2[ell][2] - z1[k][2]*z2[ell][2]) * xi[m][s,2] * v[L+2, L+2, 2, 1]
                           )
              )

        next = [q00, q10, q01, q11]
        idx = g_next[k, ell, s, m] = indmax(next)
        v_next[k, ell, s, m] = next[ idx ]
    end

    @printf("Debug: v[1,1,1,1] = %0.8f\n", v_next[1,1,1,1])
    return (v_next, g_next)
end

model = DynamicModel(bellmanUpdate; objective=:Max)
v_initial = zeros(L+2, L+2, 2, M+1)

@time (v,g) = valueIteration(model, v_initial; discount=0.9)
@printf("Value = %0.4f\n", v[1,1,1,1])

for s = 1:2
    for m = 1:5
        bottom  = vec(g[:,1,s,m])
        top     = vec(g[:,L+2,s,m])
        
        left    = vec(g[1,:,s,m])
        right   = vec(g[L+2,:,s,m])

        @printf(" ========= s = %d m = %d =============\n", s, m)
        println(rle(bottom))
        println(rle(top))
        println(rle(left))
        println(rle(right))
    end
end

