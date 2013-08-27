module MDP

    export Model,
           ProbModel,
           DynamicModel,
           bellmanUpdate,
           valueIteration,
           valueIterationBound


    abstract Model

    type ProbModel <: Model
        bellmanUpdate :: Function
        objective  :: Function
        contractionFactor :: Float64
        stateSize  :: Int32
        actionSize :: Int32

        function ProbModel(c, P; objective=:Max) 
            (n, m) = size(c)
            if size(P) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj, cmp = (objective == :Max)? (max, >) : (min, <)

                function bellman(v::Vector{Float64}; discount=1.0)
                    Q = c + discount * reshape(P * v, n, m);

                    # vUpdated = m.objective(Q, (), 2)
                    vUpdated, gOptimal = withIndex(cmp, Q)

                    return vec(vUpdated), vec(gOptimal)
                end

                # See Puterman Thm 6.6.6
                contractionFactor = 1 - sum(min(P, (), 1))
                new (bellman, obj, contractionFactor, n, m)
            end
        end
    end

    type DynamicModel <: Model
        bellmanUpdate :: Function # (valueFunction; discount=1.0) -> (valueFunction, policy)
        objective  :: Function
        contractionFactor :: Float64

        function DynamicModel(bellmanUpdate; contractionFactor=1.0, objective=:Max) 
            if objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj = (objective == :Max)? max : min
                new (bellmanUpdate, obj, contractionFactor)
            end
        end

    end

    function valueIteration(m::Model, initial_v :: Array{Float64};
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)

        scale = (discount < 1)? (1-discount)/discount : 1.0
        scaledTolerance = scale * tolerance / 2.0

        update(v) = m.bellmanUpdate(v; discount=discount)
        
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
        v += m.objective(v - v_previous)/scale

        return (v, g)
    end

    function valueIterationBound(m::Model, initial_v :: Array{Float64};
                          discount   :: Float64 = 0.95,
                          tolerance  :: Float64 = 1e-4)
        scale = (discount < 1)? (1-discount)/discount : 1.0
        scaledTolerance = scale * tolerance / 2.0

        v,_        = m.bellmanUpdate(initial_v; discount=discount)
        precision  = spanNorm(v, initial_v)

        # See Puterman Prop 6.6.5
        # We compare with zero to allow overflow errors when precision is 0.
        iterations = abs(precision)<4*eps(Float64)? 1 :
                     log( scaledTolerance/precision ) / log ( m.contractionFactor*discount ) 

        return int(iterations + 1)
    end

    # For the probability model, it is possible to automatically initialize 
    # the initial vector
    function valueIteration(m::ProbModel;
                    initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)
      
        return valueIteration(m, initial_v;
                    discount   = discount,
                    iterations = iterations,
                    tolerance  = tolerance)
    end


    function valueIterationBound(m::ProbModel;
                          initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                          discount   :: Float64 = 0.95,
                          tolerance  :: Float64 = 1e-4)

        return valueIterationBound(m, initial_v;
                    discount   = discount,
                    tolerance  = tolerance)
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
