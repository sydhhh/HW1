function [a,b,fval,e] = dualSVM(K,y,C);
    
% min -|a|_1 + a'Ka    
% sbt a'y=0
%     a>=0

n = size(K,1);
assert(size(K,1)==size(K,2));
assert(size(y,1)==n);

YY = y*y';assert(size(YY,1)==n);
H = K.*YY;
f = -ones(n,1);
Aeq = y';
beq = 0;

lb = zeros(n,1);

ub = [];
if exist('C','var');
    ub = C*ones(n,1);
end

% opts = optimset;
% %opts.TolX = 1e-10;
% %opts.TolFun = 1e-10;
% opts.LargeScale = 'off';
% [a,fval,e,ignore,mults] = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts);
[a,fval,e,ignore,mults] = quadprog(H,f',[],[],Aeq,beq,lb,ub);

b = mults.eqlin;
a = a.*y;


