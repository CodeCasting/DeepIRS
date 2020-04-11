%% Script Description and Credits
% This script implements the algorithm proposed in the paper preprint:
% Chongwen Huang, Ronghong Mo and Chau Yuen, "Reconfigurable Intelligent
% Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement
% Learning" currently available on ARXIV: https://arxiv.org/abs/2002.10072
% Accepted by IEEE JSAC special issue on Multiple Antenna Technologies for Beyond 5G

% The actor network state is the stacked vectorized form of all channels in 
% addition to previous action, transmit/received powers as will be specified shortly.
 
% The actor network action is a stacked form of the active/passive beamforming matrices.

% The reward is the resulting sum throughput.

% Both states and actions are continious here, thus using DDPG learning agent.

% Code written by Mohammad Galal Khafagy
% Postdoctoral Researcher, American University in Cairo (AUC)
% March 2020

% This paper does not take the direct link into account, but we can still 
% take it here if we have it in the dataset
% Channel model is a narrow-band model (frequency-flat channel fading)

% This code is using the Reinforcement Learning Toolbox of MATLAB
disp('---------- Running DDPG --------')

% Environment and Agent are first created, then trained over the channels.

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

obs_len = 2*(M * N_users + M * N_BS + N_BS* N_users); % Observation (state) length (multiplied by 2 to account for complex nature)
act_len = 2*(M + N_BS* N_users);                      % Action length (number of reflecting elements + size of BS beamforming matrix)

INPUT = zeros(sim_len,obs_len); % The stacked 3 vectorized channel matrices
actor_OUTPUT = zeros(sim_len, act_len); % Vectorized beamformers

%% Create Environment
% https://www.mathworks.com/help/reinforcement-learning/matlab-environments.html
% https://www.mathworks.com/help/reinforcement-learning/ug/create-matlab-environments-for-reinforcement-learning.html
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html

% The environment at a certain dtate receives the action and outputs the
% new state and the returned reward

% Our reward here in the paper is the total throughput, which we can change
% later to the (negative) sum power if needed

% ==== Specification of Observation and Action =======

% Observation (state) specification
obs_lower_lim = -Inf;
obs_upper_lim =  Inf;
obsInfo = rlNumericSpec(obs_len, 'LowerLimit', obs_lower_lim, 'UpperLimit',obs_upper_lim);
%obsInfo.Name = 'observation';
%obsInfo.Description = 'instantaneously observed channels';

% Action Specification
act_lower_lim = -Inf;
act_upper_lim =  Inf; % revise limits later for reflection coefficients
actInfo = rlNumericSpec(act_len, 'LowerLimit', act_lower_lim, 'UpperLimit',act_upper_lim);

% Step Function
% https://www.mathworks.com/help/reinforcement-learning/ug/define-reward-signals.html
%[Observation,Reward,IsDone,LoggedSignals] = stepfcn(Action,LoggedSignals)

% Reset Function
%[InitialObservation,LoggedSignals] = myResetFunction















% Create Environment
env = rlFunctionEnv(obsInfo,actInfo,stepfcn,resetfcn);

%% Create Learning Agent
% https://www.mathworks.com/help/reinforcement-learning/ug/ddpg-agents.html
% A DDPG agent consists of two agents: an actor and a critic, cooperating
% together to get a better output action

% Whitening Process for Input Decorrelation

% Define
INPUT  = [real(Ht(:)); imag(Ht(:));
    real(Hr(:)); imag(Hr(:));
    real(Hd(:)); imag(Hd(:))].';

%OUTPUT = [real(TEMP.W(:)); imag(TEMP.W(:));
%    real(TEMP.theta(:)); imag(TEMP.theta(:))].';


% 1- Create an actor using an rlDeterministicActorRepresentation object.

% 1-a) Actor Network
actor_net = [
    % INPUT Layer
    imageInputLayer([obs_len,1,1],'Name','a_input')
    % Hidden Fully Connected Layer 1 with/without Dropout
    fullyConnectedLayer(act_len,'Name','a_fully1')
    tanhLayer('Name','a_tanh1')
    %dropoutLayer(0.5,'Name','a_dropout1')
    % Batch Normalization Layer 1
    batchNormalizationLayer('Name','a_batchNorm1')
    % Hidden Fully Connected Layer 2 with/without Dropout
    fullyConnectedLayer(4*act_len,'Name','a_fully2')
    tanhLayer('Name','a_tanh2')
    %dropoutLayer(0.5,'Name','a_dropout2')
    % Batch Normalization Layer 2
    batchNormalizationLayer('Name','a_batchNorm2')
    % OUTPUT Layer
    fullyConnectedLayer(act_len,'Name','a_output')
    regressionLayer('Name','a_outReg')
    % Power and Modular Normalization Layer still
    ];

actor_obsInfo

actor_actInfo

% Create actor agent
ACTOR = rlDeterministicActorRepresentation(actor_net,...
    actor_obsInfo,...
    actor_actInfo,...
    'Observation','channels',...
    'Action','beamformers');

% Critic Network
critic_net = [
    % INPUT Layer
    imageInputLayer([obs_len+act_len,1,1],'Name','c_input')
    % Hidden Fully Connected Layer 1 with/without Dropout
    fullyConnectedLayer(obs_len+act_len,'Name','c_fully1')
    tanhLayer('Name','c_tanh1')
    %dropoutLayer(0.5,'Name','c_dropout1')
    % Batch Normalization Layer 1
    batchNormalizationLayer('Name','c_batchNorm1')
    % Hidden Fully Connected Layer 2 with/without Dropout
    fullyConnectedLayer(4*(obs_len+act_len),'Name','c_fully2')
    tanhLayer('Name','c_tanh2')
    %dropoutLayer(0.5,'Name','c_dropout2')
    % Batch Normalization Layer 2
    % batchNormalizationLayer('Name','c_batchNorm2')
    % OUTPUT Layer
    fullyConnectedLayer(1,'Name','c_output')
    regressionLayer('Name','c_outReg')
    ];

critic_obsInfo

critic_actInfo

% Create critic agent
CRITIC = rlQValueRepresentation(critic_net,...
    critic_obsInfo,...
    critic_actInfo,...
    'Observation','states_actions',...
    'Action','Q_approx');

% 3- Specify DDPG options
OPTIONS =    rlDDPGAgentOptions('DiscountFactor',gam, ...
    'ExperienceBufferLength',T,...
    'MiniBatchSize', W_exp);

% 4- Create DDPG agent
agent = rlDDPGAgent(ACTOR,CRITIC,OPTIONS);

%% Train the agent in the environment
% https://www.mathworks.com/help/reinforcement-learning/ug/train-reinforcement-learning-agents.html

% Options
train_options = trainingOptions('adam', ...       % Updated as in the paper
    'GradientDecayFactor', 0.001,...
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

