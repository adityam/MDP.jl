using MDP

const rate        = [0.3 0.7 0.9]
const arrivalRate = 0.5

const serviceCost = [6 10 20]
const holdingCost = 1
const dropPenalty = 100_000

const M = 100
const A = size(rate,2)


P = [ spzeros(Float64, M+1, M+1) for u = 1:A]
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
        P[u][x, x-1] = (1 - arrivalRate) * rate[u]
        P[u][x, x]   = (1 - arrivalRate) * (1 - rate[u]) + arrivalRate * rate[u]
        P[u][x, x+1] = arrivalRate * ( 1 - rate[u])
    end
end

for u = 1:A
    P[u][1,1] = (1 - arrivalRate) 
    P[u][1,2] = arrivalRate

    P[u][M+1, M+1] = (1 - rate[u])
    P[u][M+1, M  ] = rate[u]
end

model = ProbModel(C, P; objective=:Min)

@time (v,g) = valueIteration(model; discount=0.9)

using StatsBase

println(rle(g))
