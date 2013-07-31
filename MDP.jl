module MDP

    export Model,
           bellmanUpdate,
           valueIteration

    type Model
        stateSize  :: Int32
        actionSize :: Int32
        perStep    :: Array{Float64,2}
        transition :: Array{Float64,2}
        objective  :: Function

        function Model(n,m,c,P,sense) 
            if size(c) != (n,m) || size(P) != (n*m, n)
                error("Matrix dimensions do not match")
            elseif sense != :Max && sense != :Min 
                error("Model sense must be :Max or :Min")
            else
                obj = (sense == :Max)? max : min
                new (n,m,c,P,obj)
            end
        end
    end

    Model(c,P, sense) = Model(size(c,1),size(c,2),c,P, sense)

    function bellmanUpdate(m::Model, valueFunction::Array{Float64,1}; discount=1.0)
        Q = m.perStep + discount *
               reshape(m.transition * valueFunction, m.stateSize, m.actionSize);

        vUpdated = m.objective(Q,(),2)

        return vec(vUpdated)
    end

    function valueIteration(m::Model;
                    initial_v  :: Array{Float64,1} = vec(zeros(m.stateSize)),
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
