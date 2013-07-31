module MDP

    export Model,
           bellmanUpdate,
           valueIteration

    type Model
        stateSize  :: Int32
        actionSize :: Int32
        perStep    :: Matrix{Float64}
        transition :: Matrix{Float64}
        objective  :: Function
        findIndex  :: Function

        function Model(c,P;objective=:Max) 
            (n,m) = size(c)
            if size(P) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj,ind = (objective == :Max)? (max,indmax) : (min,indmin)
                new (n,m,c,P,obj,ind)
            end
        end
    end

    function bellmanUpdate(m::Model, valueFunction::Vector{Float64}; discount=1.0)
        Q = m.perStep + discount *
               reshape(m.transition * valueFunction, m.stateSize, m.actionSize);

        # vUpdated = m.objective(Q,(),2)
        vUpdated, gOptimal = withIndex(m.findIndex,Q)

        return vec(vUpdated), vec(gOptimal)
    end

    function valueIteration(m::Model;
                    initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)
      
        scale = (discount < 1)? (1-discount)/discount : 1.0
        scaledTolerance = scale * tolerance
        
        update(v) = bellmanUpdate(m,v; discount=discount)

        v,g        = update(initial_v)
        v_previous = copy(v)

        for i = 1:iterations
            copy!(v_previous, v)
            v,g = update(v_previous)
            
            if spanNorm(v, v_previous) < scaledTolerance
                println("Reached precision ", tolerance, " at iteration ", i)
                break
            end
        end
        
        # Renormalize v -- See Puterman 6.6.12 for details
        v += m.objective(v - v_previous)/scale

        return v,g
    end

    function spanNorm (x::Vector, y::Vector)
        z = x - y;
        return max(z) - min(z)
    end

    # Julia does not have an in-built function that returns the minimum and the
    # arg min.
    function withIndex{T}(f::Function, x::Matrix{T})
        idx = vec(mapslices(f,x,2))
        val :: Vector{T} = [x[i,idx[i]] for i = 1:size(x,1) ]

        return val,idx
    end

end
