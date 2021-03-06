clear;
close all;
load('./saved/phi_stable.mat', 'phi_stable');
load('./saved/alpha.mat','alpha');
load('./saved/gamma.mat','gamma');
load('./saved/beta.mat','beta');
sdpvar x1 x2;
vars = [x1 x2];
f = [x2;
    -x1+x2-x1^2*x2];
deg = 6;
mono = monolist(vars,deg);
M = size(phi_stable,2);
V = 0;
for i = 1:M
    V = V + alpha(i)*((mono'*real(phi_stable(:,i)))^2 + (mono'*imag(phi_stable(:,i)))^2);
end

%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%
beta = 1;
dV = jacobian(V,x1)*f(1) + jacobian(V,x2)*f(2);
[s,c] = polynomial(vars,5);
t = -(dV - beta*(gamma - V)) - s*(gamma - V);
F =  [sos(s), sos(t)];
opt = sdpsettings('solver','mosek', 'debug',1, 'showprogress', 1);
sol = solvesos(F,[],opt,c);
