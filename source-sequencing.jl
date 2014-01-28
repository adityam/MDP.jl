using MDP

p1 = 0.1; q1 = 0.3;
p2 = 0.1; q2 = 0.1;

P1 = [1 - p1 p1; q1 1 - q1];
P2 = [1 - p2 p2; q2 1 - q2];

size1 = size(P1)[1]
size2 = size(P2)[1]

M = 5

# Reachable states
pi1 = [ P1^n for n = 1:M ]
pi2 = [ P2^n for n = 1:M ]

# Distortion
d1 = 1 - eye(size1);
d2 = 1 - eye(size2);

Davg1 = [ minimum(pi1[n]*d1,1) for n = 1:M ] 
Davg2 = [ minimum(pi2[n]*d1,1) for n = 1:M ] 

function bellmanUpdate(v; discount=1.0)
    # Assume that v is size1 x size2 x size1 x M x size2 x M
    v_next = zeros(Float64, size(v))
    g_next = zeros(Int,     size(v))

    # v_next(s1, s2, z1, k1, z2, k2)

    W1 = zeros(Float64, (size1, size2, M))
    W2 = zeros(Float64, (size2, size1, M))
    
    # We go for a dumb solution to see how fast it runs.
    # If speed becomes an issue, refactor this to reduce the number of loops
    
    # Calculate W1
    for k2 = 1:M
        k2_next = min(k2+1,M)
        for z2 = 1:size2
            for s1 = 1:size1
                # Note that the cost to go is just a quadratic form
                # cost_to_go = 0 ;
                # for s2_next = 1:size2
                #     for s1_next = 1:size1
                #         cost_to_go += ( pi1[1][s1,s1_next] * pi2[k2_next][z2,s2_next]
                #                         * v(s1_next, s2_next, s1, 1, z2, k2_next)
                #                         )
                #     end
                # end
                cost_to_go = pi1[1][s1,:] * v[:,:, s1, 1, z2, k2_next] * pi2[k2_next][z2,:]'
                # In Julia 1x1 matrix is not a scalar
                W1[s1,z2,k2] = Davg2[k2][z2] + discount * cost_to_go[1,1]
            end
        end
    end

    # Calculate W2
    for k1 = 1:M
        k1_next = min(k1+1,M)
        for z1 = 1:size1
            for s2 = 1:size2
                cost_to_go = pi1[k1_next][z1,:] * v[:,:, 1, s2, z1, k1_next] * pi2[1][s2,:]'
                W2[s2,z1,k1] = Davg1[k1][z1] + discount * cost_to_go[1,1]
            end
        end
    end

    # Calculate v_next and g_next
    for k2 = 1:M
        for z2 = 1:size2
            for k1 = 1:M
                for z1 = 1:size1
                    for s2 = 1:size2
                        for s1 = 1:size1
                            # Choose the option will lower cost
                            if W1[s1, z2, k2] < W2[s2, z1, k1]
                                v_next[s1, s2, z1, k1, z2, k2] = W1[s1, z2, k2]
                                g_next[s1, s2, z1, k1, z2, k2] = 1
                            else
                                v_next[s1, s2, z1, k1, z2, k2] = W2[s2, z1, k1]
                                g_next[s1, s2, z1, k1, z2, k2] = 2
                            end
                        end
                    end
                end
            end
        end
    end

    return (v_next, g_next)

end

model = DynamicModel(bellmanUpdate; objective=:Min)
v_initial = zeros(size1, size2, size1, M, size2, M)

@time (v,g) = valueIteration(model, v_initial; discount=0.9)
