using MDP

P1 = [ 0.2 0.5 0.3; 
       0.3 0.2 0.5;
       0.5 0.3 0.2] ;

P2 = [ 0.8 0.1 0.1;
       0.1 0.8 0.1;
       0.1 0.1 0.8];

P = [P1; P2];

c = [ 1.0  2.0;
      2.0  1.0;
     10.0 10.0];

model = ProbModel(c, P; objective=:Max);
@printf("Should take atmost %d iterations\n", valueIterationBound(model))

v = valueIteration(model)
println(v)

println("Puterman's example")

P1 = [ 0.5 0.5 ;
       0.0 1.0 ];

P2 = [ 0.0 1.0 ;
       0.5 0.5 ];

c = [5.0  10.0;
     -1.0 Inf];

P = [P1; P2];

model = ProbModel(c, P; objective=:Min);
@printf("Should take atmost %d iterations\n", valueIterationBound(model))
v = valueIteration(model; discount=0.95);

println(v)

println("Puterman's example using Dynamic model")
function bellmanUpdate(v; discount=1.0)
    v_next = zero(v)
    g_next = zeros(size(v))    

    update_1 = [  5 + discount * ( 0.5*v[1] + 0.5*v[2] ),
                 10 + discount * 1.0*v[2] ]
    g_next[1] = indmin(update_1)
    v_next[1] = update_1[g_next[1]]

    # Only one alternative available
    g_next[2] = 1
    v_next[2] = -1 + discount * v[2]

    return (v_next, g_next)
end

model = DynamicModel(bellmanUpdate; objective=:Min)
v = valueIteration(model, zeros(2,1); discount=0.95);
println(v)

