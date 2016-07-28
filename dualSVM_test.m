

x = [1 0;
     0 1;
     1 1];

y = [1,-1,1]';
K = linear(x);
[alpha,b,fval,e] = dualSVM(K,y);
[w,b,fval,e] = primalSVM(x,y,1);

w_dual = (alpha'*x)';
assert( sum(abs(w-w_dual))<1e-5,'primal and dual give different solutions');

pred = sign(K*alpha + b);
assert(all(pred==y));
assert(e==1);

figure(1);clf; hold on
ind = find(y==1);
plot(x(ind,1),x(ind,2),'ro');
ind = find(y~=1);
plot(x(ind,1),x(ind,2),'bo');

xtest = [0.25,1];
assert(sign(xtest*w) == sign(alpha'*(x*xtest')));


% xor
x = [1 0;
     0 1;
     1 1;
     0 0];
y = [1,1,-1,-1]';
K = rbf(x,1);
[alpha,b,fval,e] = dualSVM(K,y);
assert(e==1);
pred = sign(K*alpha+b);
assert(all(pred==y));

x = randn(100,2);
x(1:50,:) = x(1:50,:) + 3;
x(51:end,:) = x(51:end,:) - 3;
y = ones(100,1);y(1:50)=-1;

gam = rand;
K = rbf(x,gam);
[alpha,b,fval,e] = dualSVM(K,y);
pred = sign(K*alpha+b);
assert(all(pred==y));

K = linear(x)+0.001*eye(100);
[alpha,b,fval,e] = dualSVM(K,y);
[w,b,fval,e] = primalSVM(x,y,1);
w2 = (alpha'*x)';
assert(sum(abs(w==w2))<1e-5);
pred = sign(K*alpha+b);
assert(all(pred==y));


figure(1);clf;hold on
plot(x(1:50,1),x(1:50,2),'bo');
plot(x(51:end,1),x(51:end,2),'ro');
