module MDP

    export Model,
           ProbModel,
           DynamicModel,
           valueIteration,
           finiteHorizon


    abstract Model

    type ProbModel <: Model
        bellmanUpdate :: Function
        objective  :: Function
        contractionFactor
        stateSize  :: Int
        actionSize :: Int

        function ProbModel(c, P; objective=:Max) 
            (n, m) = size(c)
            P_concatenated = vcat(P...)
            if size(P_concatenated) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif objective != :Max && objective != :Min 
                error("Model objective must be :Max or :Min")
            else
                obj, cmp = (objective == :Max)? (maximum, >) : (minimum, <)

                function bellman(v; discount=1)
                    Q = c + discount * reshape(P_concatenated * v, n, m);

                    # vUpdated = m.objective(Q, (), 2)
                    vUpdated, gOptimal = withIndex(cmp, Q)

                    return vec(vUpdated), vec(gOptimal)
                end

                # See Puterman Thm 6.6.6
                contractionFactor = 1 - sum(minimum(P_concatenated, 1))

                new (bellman, obj, contractionFactor, n, m)
            end
        end
    end

    type DynamicModel <: Model
        bellmanUpdate :: Function # (valueFunction; discount=1.0) -> (valueFunction, policy)
        objective  :: Function
        contractionFactor 

        function DynamicModel(bellmanUpdate; contractionFactor=1, objective=:Max) 
            if objective != :Max && objective != :Min 
                error("Model objective must be :Max or :Min")
            else
                obj = (objective == :Max)? maximum : minimum
                new (bellmanUpdate, obj, contractionFactor)
            end
        end

    end

    function valueIteration(m::Model, initial_v;
                    discount   = 0.95,
                    iterations = 1_000,
                    tolerance  = 1e-4)

        scale = (discount < 1)? (1-discount)/discount : 1
        scaledTolerance = scale * tolerance / 2

        update(v) = m.bellmanUpdate(v; discount=discount)
        
        (v, g)     = update(initial_v)
        v_previous = copy(v)
        precision  = spanNorm(v, initial_v)
        
        # See Puterman Prop 6.6.5
        # We compare with zero to allow overflow errors when precision is 0.
        iteration_bound = abs(precision)<4*eps(Float64)? 1 :
                     int(log( scaledTolerance/precision ) / log ( m.contractionFactor*discount ))

        info("value iteration will converge in at most $iteration_bound iterations")
        if (iterations <= iteration_bound)
            warn("Value iteration may not converge. Iterations $iterations less than estimated bound $iteration_bound")
        end 

        iterationCount  = 1;
        while (precision > scaledTolerance && iterationCount < iterations)
            iterationCount += 1

            copy!(v_previous, v)
            (v, g) = update(v_previous)

            precision = spanNorm(v, v_previous)
        end

        if (precision > scaledTolerance)
            warn(@sprintf("Value iteration did not converge. 
                 Reached precision %e at iteration %d", 2*precision/scale, iterationCount))
        else
            info(@sprintf("Reached precision %e at iteration %d", 2*precision/scale, iterationCount))
        end

        # Renormalize v -- See Puterman 6.6.12 for details
        v .+= m.objective(v - v_previous)/scale

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

        update(v) = m.bellmanUpdate(v; discount=1)

        v = [ zero(final_v) for stage = 1 : horizon ]
        g = [ zeros(Int,   size(final_v)) for stage = 1 : horizon ]

        v[horizon] = copy(final_v)
        for stage = horizon-1: -1 : 1
          (v[stage], g[stage]) = update(v[stage+1])
        end
        return (v,g)
    end
        
    function spanNorm (x, y)
        # z = x - y
        # return max(z) - min(z)
        # Optimized code. This could have been done using @devec, but
        # I don't want to add a dependency just for one function.
        z = vec(x - y)
        max_z, min_z = -Inf, Inf
        for elem = z 
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
    function withIndex{T,N} (compare::Function, x::AbstractArray{T,N})
        (n, m) = size(x)
        idx = vec(zeros(Int, n, 1))
        val = vec(zeros(T,   n, 1))

        for i=1:n
            idx[i], val[i] = 1, x[i,1]
            for j=2:m
                if compare(x[i,j], val[i])
                    idx[i], val[i] = j, x[i,j]
                end
            end
        end

        return (val, idx)
    end
end
