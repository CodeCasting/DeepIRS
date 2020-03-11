function [ACTOR,CRITIC,OPTIONS,agent, W, theta] = drl_IRS(P_t,Ht,Hr,Hd)

%% Function Documentation
% This function implements the algorithm proposed in the paper preprint:

% Chongwen Huang, Ronghong Mo and Chau Yuen, "Reconfigurable Intelligent
% Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement
% Learning" currently available on ARXIV: https://arxiv.org/abs/2002.10072
% Accepted by IEEE JSAC special issue on Multiple Antenna Technologies for Beyond 5G

% The actor network state is the stacked vectorized form of all channels.
% The actor network action is a stacked form of the active/passive beamforming matrices.
% The reward is the resulting sum throughput.

disp('---------- Running DDPG --------')

%% Simulation Parameters (in Table 1 in the paper)

% For more info about DDPG agent in MATLAB, see:
% https://www.mathworks.com/help/reinforcement-learning/ug/ddpg-agents.html#mw_086ee5c6-c185-4597-aefc-376207c6c24c
% For other supported Reinforcement Learning agents see:
% https://www.mathworks.com/help/reinforcement-learning/ug/create-agents-for-reinforcement-learning.html

% Actor
u_a = 1e-3;     % learning rate for training actor network uptate
lam_a= 1e-5;    % decaying rate for training actor network uptate
% Target Actor
t_a = 1e-3;     % learning rate for target actor network uptate
% Critic
u_c = 1e-3;     % learning rate for training critic network uptate
lam_c= 1e-5;    % decaying rate for training critic network uptate
% Target Critic
t_c = 1e-3;     % learning rate for target critic network uptate

gam = 0.99;     % Discount factor
D = 1e5;        % Length of memory window
N_epis = 5e3;   % Number of episodes (changed due to DUPLICATE NAME)
T = 2e4;        % Number of steps per episode
W_exp = 16;     % Number of experiences in the mini-batch
U = 1;          % Number of steps synchronizing target with training network

%% Memory Preallocation
N_users = size(Hr,2);
M = size(Ht,1);
N_BS = size(Ht,2);

INPUT = zeros(sim_len,2*(M * N_users + M * N_BS + N_BS* N_users)); % The 3 vectorized channel matrices
actor_OUTPUT = zeros(sim_len, 2*(M + N_BS* N_users)); % Vectorized beamformers

%% Define Actor/Critic NN Layers and Agent

% Whitening Process for Input Decorrelation 

% Define 
INPUT  = [real(Ht(:)); imag(Ht(:));
    real(Hr(:)); imag(Hr(:));
    real(Hd(:)); imag(Hd(:))].';
%OUTPUT = [real(TEMP.W(:)); imag(TEMP.W(:));
%    real(TEMP.theta(:)); imag(TEMP.theta(:))].';

% Actor Network
actor_layers = [
    % INPUT Layer
    imageInputLayer([length(INPUT),1,1],'Name','a_input')
    
    % Hidden Fully Connected Layer 1 with/without Dropout
    fullyConnectedLayer(length(actor_OUTPUT),'Name','a_fully1')
    tanhLayer('Name','a_tanh1')
    %dropoutLayer(0.5,'Name','a_dropout1')
    
    % Batch Normalization Layer 1
    batchNormalizationLayer('Name','a_batchNorm1')
    
    % Hidden Fully Connected Layer 2 with/without Dropout
    fullyConnectedLayer(4*length(actor_OUTPUT),'Name','a_fully2')
    tanhLayer('Name','a_tanh2')
    %dropoutLayer(0.5,'Name','a_dropout2')
    
    % Batch Normalization Layer 2
    batchNormalizationLayer('Name','a_batchNorm2')
    
    % OUTPUT Layer
    fullyConnectedLayer(length(actor_OUTPUT),'Name','a_output')
    regressionLayer('Name','a_outReg')
    
    % Power and Modular Normalization Layer
    ];

% Critic Network
critic_layers = [
    % INPUT Layer
    imageInputLayer([length(actor_OUTPUT),1,1],'Name','c_input')
    
    % Hidden Fully Connected Layer 1 with/without Dropout
    fullyConnectedLayer(length(actor_OUTPUT),'Name','c_fully1')
    tanhLayer('Name','c_tanh1')
    %dropoutLayer(0.5,'Name','c_dropout1')
    
    % Batch Normalization Layer 1
    batchNormalizationLayer('Name','c_batchNorm1')
    
    % Hidden Fully Connected Layer 2 with/without Dropout
    fullyConnectedLayer(4*length(actor_OUTPUT),'Name','c_fully2')
    tanhLayer('Name','c_tanh2')
    %dropoutLayer(0.5,'Name','c_dropout2')
    
    % Batch Normalization Layer 2
    % batchNormalizationLayer('Name','c_batchNorm2')
    
    % OUTPUT Layer
    fullyConnectedLayer(length(actor_OUTPUT),'Name','c_output')
    regressionLayer('Name','c_outReg')
    
    ];

% Options
options = trainingOptions('adam', ...       % Updated as in the paper
    'MiniBatchSize',W_exp, ...              % Updated as in the paper
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',3, ...
    'L2Regularization',1e-4,...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','none', ... % 'training-progress'
    'Verbose',0, ...    % 1
    'ExecutionEnvironment', 'auto', ...
    'VerboseFrequency',VerboseFrequency);

% Adjust DDPG options
OPTIONS = rlDDPGAgentOptions('DiscountFactor',gam, ...
    'ExperienceBufferLength',T,...
    'MiniBatchSize', W_exp);

% Create DDPG agent
agent = rlDDPGAgent(ACTOR,CRITIC,OPTIONS);


end

