clc
clear
close all;

% Multi-objective function (Choose among functions)
MultiObjFnc = 'Random Forest_MOPSO';

switch MultiObjFnc
     
     case  'Random Forest_MOPSO'            %Random Forest
        load('PF.mat');
        load data.csv;
        X = data (:,1:5);
        ROP = data (:,6);
        TOB = data (:,7);
        template = templateTree('MinLeafSize', 3);
        f1 = @(x,y,a,b,c) -oobPredict(fitrensemble(X, ROP, 'Method', 'Bag','NumLearningCycles', 50,'Learners', template));
        f2 = @(x,y,a,b,c) oobPredict(fitrensemble(X, TOB, 'Method', 'Bag','NumLearningCycles', 50,'Learners', template));
        
        MultiObj.fun = @(x) [f1(x(:,1),x(:,2),x(:,3),x(:,4),x(:,5)), f2(x(:,1),x(:,2),x(:,3),x(:,4),x(:,5))];
        MultiObj.nVar = 5;
        MultiObj.var_min = [2229, 164, 448.5, 322.91, 1038.9];
        MultiObj.var_max = [17070, 181.99, 502.4, 330, 1349.11];
        MultiObj.truePF = PF;
    
end

% Parameters
params.Np = 200;        % Population size
params.Nr = 112;        % Repository size
params.maxgen = 100;    % Maximum number of generations
params.W = 0.4;         % Inertia weight
params.C1 = 2;          % Individual confidence factor
params.C2 = 2;          % Swarm confidence factor
params.ngrid = 20;      % Number of grids in each dimension
params.maxvel = 5;      % Maxmium vel in percentage
params.u_mut = 0.5;     % Uniform mutation percentage

% MOPSO
REP = MOPSO(params,MultiObj);

% Display info
display('Repository fitness values are stored in REP.pos_fit');
display('Repository particles positions are store in REP.pos');


    