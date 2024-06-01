%----- Kernel-based Meshless Method of Lines -----%
%------------------ Example 2 --------------------%
%----- Uniform Nodes over Recatngular Domain -----%
warning off; clear all; close all; clc; format long e;
rbf = input(' RBF KERNEL = '); % IMQ = 1, GA = 2
tic;
dd=2; % parameter dd=1 (full), dd=4,2 (reduced)
alpha=1; beta=1; gama=1/1000; % problem parameters 
t0=0; tf=01; % initial and final time
dt=1/100; nf=tf/dt; % time-step size and max number of time-steps
a=0; b=1; N1=32; N=N1^2; % number of collocation points
M1=round(sqrt((1/dd)*N)); M=M1*M1; % number of center points
if rbf == 1
c=0.007*sqrt(M*N); % shape parameter
phai=@(r,s)     1./sqrt(1 + (r.*s).^2); % IMQ 
phai2=@(r,rh,s) (3*s.^4*(2*rh).^2)./(4*((r.*s).^2 + 1).^(5/2)) ...
                - s^2./(((r.*s).^2 + 1).^(3/2));
elseif rbf == 2
c=0.03*sqrt(M*N); % shape parameter
phai=@(r,s)     exp(-(s*r).^2); % GA 
phai2=@(r,rh,s) s^4*exp(-(s*r).^2).*(2*rh).^2 - 2*s^2*exp(-(s*r).^2);
else 
    fprintf('invalid input')
    return
end
%- functions -%
uf = @(X,Y)     exp(-tf).*exp(-(X-0.5).^2/gama-(Y-0.5).^2/gama); % final sol
uIC  = @(X,Y)   exp(-t0).*exp(-(X-0.5).^2/gama-(Y-0.5).^2/gama); % initial sol
uEx =@(X,Y,T)   exp(-T).*exp(-(X-0.5).^2/gama-(Y-0.5).^2/gama); % exact sol
duEx =@(X,Y,T) -exp(-T).*exp(-(X-0.5).^2/gama-(Y-0.5).^2/gama); % fb'(t)
Ff =@(X,Y,T)   -exp(-T).*exp(-(X-0.5).^2/gama - (Y-0.5).^2/gama);  % source
gridCtrs  = linspace(a,b,M1); [xc,yc]=meshgrid(gridCtrs); % Center set
gridData  = linspace(a,b,N1); [xd,yd]=meshgrid(gridData); % Collocation set
Ctrs=[xc(:) yc(:)]; Data=[xd(:) yd(:)];
indexBC=find(Ctrs(:,1)==a | Ctrs(:,1)==b | Ctrs(:,2)==a | Ctrs(:,2)==b); 
bCdata=Ctrs(indexBC,:);
indexIC=find(Ctrs(:,1)~=a & Ctrs(:,1)~=b & Ctrs(:,2)~=a & Ctrs(:,2)~=b); 
intCdata=Ctrs(indexIC,:); 
indexBD=find(Data(:,1)==a | Data(:,1)==b | Data(:,2)==a | Data(:,2)==b); 
bDdata=Data(indexBD,:);
indexID=find(Data(:,1)~=a & Data(:,1)~=b & Data(:,2)~=a & Data(:,2)~=b); 
intDdata=Data(indexID,:); 
figure;
plot(intDdata(:,1),intDdata(:,2),'b^','LineWidth',1.5)
hold on
plot(bDdata(:,1),bDdata(:,2),'b*','LineWidth',1.5)
hold on
plot(intCdata(:,1),intCdata(:,2),'ro','LineWidth',1.5);
hold on
plot(bCdata(:,1),bCdata(:,2),'rs','LineWidth',1.5); axis([a-0.5 b+0.5 a-0.5 b+0.5])
xC=Ctrs(:,1); yC=Ctrs(:,2); xD=Data(:,1); yD=Data(:,2);
%%--------------RBF Matrices-----------%%
r=zeros(N,M); rx=zeros(N,M); ry=zeros(N,M); 
for i=1:N
    for j=1:M
        rx(i,j)=(xD(i)-xC(j)); ry(i,j)=(yD(i)-yC(j));
        r(i,j)=sqrt(rx(i,j).^2+ry(i,j).^2);
    end
end
%- RBF Matrice & Decomposition -%
A11=phai(r(indexID,indexIC),c); A22=phai(r(indexBD,indexBC),c); 
A12=phai(r(indexID,indexBC),c); A21=phai(r(indexBD,indexIC),c);  
D11=phai2(r(indexID,indexIC),rx(indexID,indexIC),c) ...
    + phai2(r(indexID,indexIC),ry(indexID,indexIC),c);
D22=phai2(r(indexBD,indexBC),rx(indexBD,indexBC),c) ...
    + phai2(r(indexBD,indexBC),ry(indexBD,indexBC),c); 
D12=phai2(r(indexID,indexBC),rx(indexID,indexBC),c) ...
    + phai2(r(indexID,indexBC),ry(indexID,indexBC),c); 
D21=phai2(r(indexBD,indexIC),rx(indexBD,indexIC),c) ...
    + phai2(r(indexBD,indexIC),ry(indexBD,indexIC),c);   
%-------- QR Decomposition ----------%
[Q1,R1,E1]=qr(A11); [Q2,R2,E2]=qr(A22); 
invAi=E1*(R1\Q1'); invAb=E2*(R2\Q2'); 
%- Perparation for time advancement -%
u0I=uIC(intDdata(:,1),intDdata(:,2)); 
u0B=uIC(bDdata(:,1),bDdata(:,2));    
lambdaI=invAi*u0I; % lamda_I
AL=A11-A12*invAb*A21-alpha*D11+alpha*D12*invAb*A21; % LHS matrix
AR=beta*D11-beta*D12*invAb*A21; % RHS matrix
[QL,RL,EL]=qr(AL); invAL=EL*(RL\QL');
ER=invAL*AR; % RHS matrix
FI=@(X,Y,T) invAL*Ff(X,Y,T); % interior RHS function
FB=@(X,Y,T) invAL*(-A12*invAb*duEx(X,Y,T) ...
            + alpha*D12*invAb*duEx(X,Y,T) ...
            + beta*D12*invAb*uEx(X,Y,T)); % Boundary RHS function
rhsF=@(T,V) ER*V; % RHS function f(x,y,t)
%- Time loop for SSP-RK33 -%
for n=1:nf
    tn=(n-1)*dt;
    k1 = lambdaI + dt*(feval(rhsF,tn,lambdaI) ...
        + feval(FI,intDdata(:,1),intDdata(:,2),tn) ...
        +feval(FB,bDdata(:,1),bDdata(:,2),tn)); 
    k2 = (3/4)*lambdaI + (1/4)*k1+(dt/4)*(feval(rhsF,tn+dt,k1) ...
        + feval(FI,intDdata(:,1),intDdata(:,2),tn+dt) ...
        + feval(FB,bDdata(:,1),bDdata(:,2),tn+dt));
    lambdaI = (1/3)*lambdaI + (2/3)*k2+(2*dt/3)*(feval(rhsF,tn+dt/2,k2) ...
        + feval(FI,intDdata(:,1),intDdata(:,2),tn+dt/2)...
        + feval(FB,bDdata(:,1),bDdata(:,2),tn+dt/2));
end
lambda(indexBC,1)=invAb*(uEx(bDdata(:,1),bDdata(:,2),tn)-A21*lambdaI); % lamda_B
lambda(indexIC,1)=lambdaI;  % lambda_I
A=phai(r,c); % Interpolation
u=A*lambda; % sol at time t
RT=toc;
%%.............plots......................%%
uF=uf(xD,yD); 
utF=reshape(uF,N1,N1); uA=reshape(u,N1,N1);
td = delaunay(xd,yd); 
error=reshape(abs(u-uF),N1,N1);
figure;
subplot(1,2,1)
trimesh(td,xd,yd,uA), xlabel x,ylabel y, zlabel uA; title(sprintf('Approx at t = %1.1f',tf)), colorbar; colormap jet
subplot(1,2,2)
trisurf(td,xd,yd,error), xlabel x,ylabel y, zlabel AE; title(sprintf('Error at t = %1.1f',tf)), colorbar; colormap jet
figure;
plot(real(eig(ER)),imag(eig(ER)),'*'); xlabel Re(z),ylabel Im(z); title('unscaled eigenvalues'), colormap jet;
format short e;
errInf=norm(u-uF,inf); RMS=norm(u-uF,2)/sqrt(N);
[errInf RMS RT]
%----------------------- THE EnD ----------------------%