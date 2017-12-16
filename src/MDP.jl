module MDP

    export Model,
           ProbModel,
           DynamicModel,
           valueIteration,
           finiteHorizon


    abstract type Model end

    type ProbModel <: Model
        bellmanUpdate! :: Function
        objective  :: Function
        contractionFactor :: Float64
        stateSize  :: Int
        actionSize :: Int

        P :: Array{ Matrix{Float64}, 1}
        cost   :: Matrix{Float64}
        reward :: Matrix{Float64}

        function ProbModel(c, P; objective=:Max) 
            (n, m) = size(c)

            if length(P) != m
                error("Number of transition matrices does not match number of actions.")
            end

            P_concatenated = vcat(P...)
            if size(P_concatenated) != (n*m, n)
                error("Size of transition and reward matrices are inconsistent")
            end

            is_square(Pi)          = size(Pi) == (n,n)
            is_row_stochastic(Pi)  = isapprox(sum(Pi, 2), ones(n); atol = 100*eps(Float64))
            is_stopping_action(Pi) = Pi == zero(Pi)
            for Pi in P
                if !is_square(Pi)
                    error("Transition matrix is not a square matrix")
                elseif !(is_row_stochastic(Pi) || is_stopping_action(Pi))
                    error("Transition matrix is not row stochastic.")
                end
            end

            if objective != :Max && objective != :Min 
                error("Model objective must be :Max or :Min")
            else
                obj, compare = (objective == :Max)? (maximum, >) : (minimum, <)

                function bellman!(vUpdated, gUpdated, v; discount=1)
                    Q = c + discount * reshape(P_concatenated * v, n, m);

                    # vUpdated = m.objective(Q, (), 2)
                    withIndex!(vUpdated, gUpdated, compare, Q)
                end

                # See Puterman Thm 6.6.6
                contractionFactor = 1 - sum(minimum(P_concatenated, 1))

                new(bellman!, obj, contractionFactor, n, m, P, c, c)
            end
        end
    end

    type DynamicModel <: Model
        bellmanUpdate! :: Function # (valueFunction; discount=1.0) -> (valueFunction, policy)
        objective  :: Function
        contractionFactor :: Float64

        function DynamicModel(bellmanUpdate!; contractionFactor=1, objective=:Max) 
            if objective != :Max && objective != :Min 
                error("Model objective must be :Max or :Min")
            else
                obj = (objective == :Max)? maximum : minimum
                new(bellmanUpdate!, obj, contractionFactor)
            end
        end

    end

    function valueIteration(m::Model, initial_v;
                    discount   = 0.95,
                    iterations = 1_000,
                    tolerance  = 1e-4)

        scaledDiscount  = (discount < 1)? (1-discount)/discount : 1
        scaledTolerance = scaledDiscount * tolerance / 2

        update!(vUpdated, gUpdated, v) = m.bellmanUpdate!(vUpdated, gUpdated, v; discount=discount)

        v_previous = copy(initial_v)
        v          = copy(initial_v)
        g          = zeros(Int, size(v))
        
        update!(v, g, initial_v)
        copy!(v_previous, v)

        v_precision = spanNorm(v, initial_v)
        
        if abs( 1 - m.contractionFactor * discount ) < 4*eps(Float64)
            warn("Contraction factor too small to guarantee convergece. Value iteration may not converge.")
        else
            # See Puterman Prop 6.6.5
            # We compare with zero to allow overflow errors when v_precision is 0.
            if abs(v_precision) < 4*eps(Float64)
                iteration_bound = 1
            else
                iteration_bound = floor(Int, log( scaledTolerance/v_precision ) / log( m.contractionFactor*discount ))
            end

            info("value iteration will converge in at most $iteration_bound iterations")
            if (iterations <= iteration_bound)
                warn("Value iteration may not converge. Iterations $iterations less than estimated bound $iteration_bound")
            end 
        end

        iterationCount  = 1;
        while (v_precision > scaledTolerance && iterationCount < iterations)
            iterationCount += 1

            copy!(v_previous, v)
            update!(v, g, v_previous)

            v_precision = spanNorm(v, v_previous)
        end

        if (v_precision > scaledTolerance)
            warn(@sprintf("Value iteration did not converge. 
                 Reached precision %e at iteration %d", 2*v_precision/scaledDiscount, iterationCount))
        else
            info(@sprintf("Reached precision %e at iteration %d", 2*v_precision/scaledDiscount, iterationCount))
        end

        # Renormalize v -- See Puterman 6.6.12 for details
        v .+= m.objective(v - v_previous)/scaledDiscount

        return (v, g)
    end

    # For the probability model, it is possible to automatically initialize 
    # the initial vector
    function valueIteration(m::ProbModel;
                    initial_v  = zeros(m.stateSize),
                    discount   = 0.95,
                    iterations = 1_000,
                    tolerance  = 1e-4)
      
        return valueIteration(m, initial_v;
                    discount   = discount,
                    iterations = iterations,
                    tolerance  = tolerance)
    end

    function finiteHorizon(m::Model, final_v;
                           horizon :: Int = 10)

        update!(vUpdated, gUpdated, v) = m.bellmanUpdate!(vUpdated, gUpdated, v; discount=discount)

        v = [ zero(final_v) for stage = 1 : horizon ]
        g = [ zeros(Int,   size(final_v)) for stage = 1 : horizon ]

        v[horizon] = copy(final_v)
        for stage in horizon-1: -1 : 1
          update!(v[stage], g[stage], v[stage+1])
        end
        return (v,g)
    end
        
    function spanNorm(x, y)
        # z = x - y
        # return max(z) - min(z)
        # Optimized code. This could have been done using @devec, but
        # I don't want to add a dependency just for one function.
        z = vec(x - y)
        max_z, min_z = -Inf, Inf
        for elem in z 
            if max_z < elem
                max_z = elem
            end
            if min_z > elem
                min_z = elem
            end
        end
        return max_z - min_z
    end

    # Julia does not have an in-built function that returns the minimum and the
    # arg min.
    # function withIndex{T}(f::Function, x::Matrix{T})
    #     idx = vec(mapslices(f,x,2))
    #     val :: Vector{T} = [x[i,idx[i]] for i = 1:size(x,1) ]

    #     return val,idx
    # end

    # A more direct implementation
    function withIndex!{T}(val::AbstractArray{T,1}, idx::AbstractArray{Int,1}, compare::Function, x::AbstractArray{T,2})
        (n, m) = size(x)

        for i=1:n
            idx[i], val[i] = 1, x[i,1]
            for j=2:m
                if compare(x[i,j], val[i])
                    idx[i], val[i] = j, x[i,j]
                end
            end
        end
    end
end
