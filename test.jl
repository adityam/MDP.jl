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

