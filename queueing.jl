using MDP

const rate        = [0.4 0.7 0.9]
const arrivalRate = 0.5

const serviceCost = [1   3   5]
const holdingCost = 9
const dropPenalty = 300

const M = 30
const A = size(rate,2)


PP = zeros(Float64, M+1, A, M+1)
C = zeros(Float64, M+1, A)

# Initialize cost matrix
C[1,:] = 0 

for x = 2:M
    for u = 1:A
        C[x,u] = (x-1) * holdingCost + serviceCost[u] 
    end
end

# Add expected cost for dopping packets
for u = 1:A 
    C[M+1,u] = M * holdingCost + serviceCost[u] + arrivalRate * dropPenalty
end

# Initialize Probability matrix
for x = 2:M
    for u = 1:A
        PP[x, u, x-1] = (1 - arrivalRate) * rate[u]
        PP[x, u, x]   = (1 - arrivalRate) * (1 - rate[u]) + arrivalRate * rate[u]
        PP[x, u, x+1] = arrivalRate * ( 1 - rate[u])
    end
end

for u = 1:A
    PP[1,u,1] = (1 - arrivalRate) 
    PP[1,u,2] = arrivalRate

    PP[M+1, u, M+1] = (1 - rate[u])
    PP[M+1, u, M  ] = rate[u]
end

# The dimensions of PP are chosen such that the following give a 
# vertical concat of the appropriate cost matrices
P = reshape(PP, ((M+1)*A, M+1) )

model = ProbModel(C, P; objective=:Min)

@time (v,g) = valueIteration(model; discount=0.5)
