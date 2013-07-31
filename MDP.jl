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

        function Model(c,P;objective=:Max) 
            (n,m) = size(c)
            if size(P) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif objective != :Max && objective != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj = (objective == :Max)? max : min
                new (n,m,c,P,obj)
            end
        end
    end

    function bellmanUpdate(m::Model, valueFunction::Vector{Float64}; discount=1.0)
        Q = m.perStep + discount *
               reshape(m.transition * valueFunction, m.stateSize, m.actionSize);

        vUpdated = m.objective(Q,(),2)

        return vec(vUpdated)
    end

    function valueIteration(m::Model;
                    initial_v  :: Vector{Float64} = vec(zeros(m.stateSize)),
                    discount   :: Float64 = 0.95,
                    iterations :: Int32   = 1_000,
                    tolerance  :: Float64 = 1e-4)
      
        scale = (discount < 1)? (1-discount)/discount : 1.0
        scaledTolerance = scale * tolerance
        
        update(v) = bellmanUpdate(m,v; discount=discount)

        v          = update(initial_v)
        v_previous = copy(v)

        for i = 1:iterations
            copy!(v_previous, v)
            v = update(v_previous)
            
            if spanNorm(v, v_previous) < scaledTolerance
                println("Reached precision ", tolerance, " at iteration ", i)
                break
            end
        end
        
        # Renormalize v -- See Puterman 6.6.12 for details
        v += m.objective(v - v_previous)/scale

        return v
    end

    function spanNorm (x::Vector, y::Vector)
        z = x - y;
        return max(z) - min(z)
    end

end
