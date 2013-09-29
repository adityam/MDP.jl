using MDP

# Only needed for rle function
using Stats

L = 100
alpha = [0.75 0.75]

P = [ alpha[1]		1-alpha[1]
	  1 - alpha[2]  alpha[2]  ]

# Initialize xi (could be optimized)
xi = [ P^n for n = 1:L+1] 

r = 1.0

for c = [0.1 0.2 0.3 0.4 0.5]
    for p = [0.1 0.2 0.3 0.4]
        q = 1 - p

        function bellmanUpdate(v; discount=1.0)
            # Assume v is 2 x 2 x (L+1)
            v_next = zeros(Float64, size(v))
            g_next = zeros(Int32,   size(v))

            # v_next(n, s, l)
            # n = 1 == 0 packets in buffer
            # n = 2 == 1 packet in buffer

            # s is the last channel state observed
            # s = 1 == Idle
            # s = 2 == Busy

            # ell is the time when the last observation was made
            for s = 1:2 
                for ell = 1:L+1
                    next_ell = min(ell+1, L+1)
                    v_next[1, s, ell] = discount * (q * v[1, s, next_ell] + p * v[2, s, next_ell])
                    g_next[1, s, ell] = 1

                    next = [ discount * v[2, s, next_ell],
                             xi[ell][s,1]*r - c + discount * ( xi[ell][s,1] * ( q * v[1, 1, 1] + p * v[2, 1, 1])
                                                             + xi[ell][s,2] * v[2, 2, 1]) ] 
                    idx = g_next[2, s, ell] = indmax(next)
                    v_next[2, s, ell] = next[ idx ] 
                end
            end

            return (v_next, g_next)
        end

        model = DynamicModel(bellmanUpdate; objective=:Max)
        v_initial = zeros(2,2,L+1)

        # @printf("Should take atmost %d iterations\n", valueIterationBound(model, v_initial; discount=0.9))

        (v,g) = valueIteration(model, v_initial; discount=0.9)

        g1 = vec(g[2,1,:])
        g2 = vec(g[2,2,:])

        a1,b1 = rle(g1[1:end-1])
        a2,b2 = rle(g2[1:end-1])

        k1 = (a1[1] == 1.0)?(b1[1]+1):1
        k2 = (a2[1] == 1.0)?(b2[1]+1):1

        @printf("c=%0.2f\tp=%0.2f\t(%d,%d)\n", c,p, k1,k2)
  end
end

# v1 = vec(v[2,1,:])
# v2 = vec(v[2,2,:])
# 
# println(v1)
# println(v2)
