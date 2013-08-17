module MDP

    export ProbModel,
           DynamicModel,
           bellmanUpdate,
           valueIteration,
           valueIterationBound

    type ProbModel
        stateSize  :: Int32
        actionSize :: Int32
        perStep    :: Matrix{Float64}
        transition :: Matrix{Float64}
        objective  :: Function
        compare    :: Function

        function ProbModel(c, P; objective=:Max) 
            (n, m) = size(c)
            if size(P) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj, cmp = (objective == :Max)? (max, >) : (min, <)
                new (n, m, c, P, obj, cmp)
            end
        end
    end

    type DynamicModel
        bellmanUpdate :: Function # (valueFunction; discount=1.0) -> (valueFunction, policy)
        objective  :: Function

        function DynamicModel(bellmanUpdate; objective=:Max) 
            if objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj = (objective == :Max)? max : min
                new (bellmanUpdate, obj)
            end
        end
    end

    function bellmanUpdate(m::ProbModel, valueFunction::Vector{Float64}; discount=1.0)
        Q = m.perStep + discount *
               reshape(m.transition * valueFunction, m.stateSize, m.actionSize);

        # vUpdated = m.objective(Q, (), 2)
        vUpdated, gOptimal = withIndex(m.compare, Q)

        return vec(vUpdated), vec(gOptimal)
    end

    function valueIteration(m::ProbModel;
                    initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)
      
        scale = (discount < 1)? (1-discount)/discount : 1.0
        update(v) = bellmanUpdate(m,v; discount=discount)

        return valueUpdateInnerLoop(initial_v, update, m.objective;
                    iterations = iterations,
                    scale      = scale,
                    tolerance  = tolerance)

    end

    function valueIteration(m::DynamicModel, initial_v :: Array{Float64};
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)

        scale = (discount < 1)? (1-discount)/discount : 1.0
        update(v) = m.bellmanUpdate(v; discount=discount)

        return valueUpdateInnerLoop(initial_v, update, m.objective;
                    iterations = iterations,
                    scale      = scale,
                    tolerance  = tolerance)
                    
    end

    function valueIterationBound(m::ProbModel;
                          initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                          discount   :: Float64 = 0.95,
                          tolerance  :: Float64 = 1e-4)

        scale = (discount < 1)? (1-discount)/discount : 1.0
        scaledTolerance = scale * tolerance / 2.0

        # See Puterman Thm 6.6.6
        k = 1 - sum(min(m.transition, (), 1))

        v,_        = bellmanUpdate(m, initial_v; discount=discount)
        precision  = spanNorm(v, initial_v)

        # See Puterman Prop 6.6.5
        iterations = log( scaledTolerance/precision ) / log ( k*discount ) 

        return int(iterations + 1)
    end
        
    function valueUpdateInnerLoop(initial_v, update, objective; 
                    iterations :: Int32   = 1_000,
                    scale      :: Float64 = 1e-2,
                    tolerance  :: Float64 = 1e-4)

        scaledTolerance = scale * tolerance / 2.0
        
        (v, g)     = update(initial_v)
        v_previous = copy(v)
        precision  = Inf32
        
        iterationCount  = 0;
        while (precision > scaledTolerance && iterationCount < iterations)
            iterationCount += 1

            copy!(v_previous, v)
            (v, g) = update(v_previous)

            precision = spanNorm(v, v_previous)
        end
        
        @printf("Reached precision %e at iteration %d\n", 2.0*precision/scale, iterationCount)

        # Renormalize v -- See Puterman 6.6.12 for details
        v += objective(v - v_previous)/scale

        return (v, g)
    end

    function spanNorm{T <: Real} (x::Array{T}, y::Array{T})
        z = x - y;
        return max(z) - min(z)
    end

    # Julia does not have an in-built function that returns the minimum and the
    # arg min.
    # function withIndex{T}(f::Function, x::Matrix{T})
    #     idx = vec(mapslices(f,x,2))
    #     val :: Vector{T} = [x[i,idx[i]] for i = 1:size(x,1) ]

    #     return val,idx
    # end

    # A more direct implementation
    function withIndex{T <: Real} (compare::Function, x::Matrix{T})
        (n, m) = size(x)
        idx::Vector{Int32} = vec(zeros(n,1))
        val::Vector{T}     = vec(zeros(n,1))

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
