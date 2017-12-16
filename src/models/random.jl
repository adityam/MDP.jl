using StatsBase, Distributions
using MDP

# Based on the description in Sec 8 of the following Technical Report
# > Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee
# > "Natural Actor–Critic Algorithms",
# > University of Alberta Department of Computing Science Technical Report TR09-10
# > June 2009.

# In the description of Garnet, there is a parameter τ that determines
# the non-stationarity of the mode. Here, we only implement the case when
# τ = 0. We will also assume that b << n, so we generate a sparse P matrix.

# The description of the garnet model says that the rewards should have normal
# distribution. We let the user to explicitly specify the reward distribution. 

function garnet(n, m, b; rewards=Uniform(0,1), objective=:Max)
   P = SparseMatrixCSC[ spzeros(n,n) for a in 1:m ]
   S = 1:n
   for a in 1:m
       for i in 1:n 
           nextState = sample(S, b; replace=false)
           nextProb  = rand(b)
           nextProb /= sum(nextProb)
           P[a][i, nextState] = nextProb
       end
   end
   c = rand(rewards, n,m)
   model = ProbModel(c, P; objective=objective)
end

function randomPolicy(model :: ProbModel; stochastic = true)
  (n, m) = (model.stateSize, model.actionSize)

  Q = Matrix[zero(model.P[1])]
  c = zeros(n,1)

  if !stochastic 
    a = sample(1:m, n; replace=true)
    for i in 1:n
      Q[1][i,:] = model.P[a[i]][i,:]
      c[i,1] = model.cost[i, a[i]]
    end
  else
    for i in 1:n
      p  = rand(m)
      p /= sum(p)

      for a in 1:m
        Q[1][i,:] += model.P[a][i,:]*p[a]
        c[i,1]    += model.cost[i,a]*p[a]
      end
    end
  end

  return ProbModel(c, Q)
end
