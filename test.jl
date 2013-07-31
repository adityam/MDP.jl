require("MDP")

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

model = Model(c,P,:Max);

v = valueIteration(model)

print(v)

