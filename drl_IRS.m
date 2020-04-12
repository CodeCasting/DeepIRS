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

% ---------- For Separate ACTOR/CRITIC AGENTS -----------------
used_optmzr = 'adam';   % Used optimizer
used_device = 'gpu';    % gpu or cpu
% Learning and Decay rates
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

% ------------- Created DDPG AGENT Options -------------------
D = 1e5;        % Length of replay experience memory window
W_exp = 16;     % Number of experiences in the mini-batch
gam = 0.99;     % Discount factor
U = 1;          % Number of steps synchronizing target with training network

% ------------- For DDPG AGENT Training ----------------------
N_epis = 5e3;           % Number of episodes (changed due to DUPLICATE NAME)
T = 2e4;                % Number of steps per episode

%% Memory Preallocation
N_users = size(Hr,2);
M = size(Ht,1);
N_BS = size(Ht,2);
% Channel Observations
chan_obs_len = 2*(M * N_users + M * N_BS + N_BS* N_users); % channel observation (state) length (multiplied by 2 to account for complex nature)
% Action length (number of reflecting elements + size of BS beamforming matrix)
% multiplied by 2 for complex nature
act_len = 2*(M + N_BS* N_users);     

transmit_pow_len = 2*N_users;
receive_pow_len = 2*N_users^2;
obs_len = transmit_pow_len + receive_pow_len + act_len + chan_obs_len;  

%% Create Environment
% https://www.mathworks.com/help/reinforcement-learning/matlab-environments.html
% https://www.mathworks.com/help/reinforcement-learning/ug/create-matlab-environments-for-reinforcement-learning.html
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html

% The environment at a certain state receives the action and outputs the
% new state and the returned reward

% Our reward here in the paper is the total throughput, which we can change
% later to the (negative) sum power if needed

% ==== Specification of Observation and Action =======
% Observation (state) specification
% State at time t is specified in the paper to be composed as follows:
% 1- Transmission power in the t^{th} time step. (Real and Imaginary power are separated)
% 2- The received power of all users in the t^{th} time step.
% 3- The previous action in the (t-1)^{th} time step.
% 4- The channels.
obs_lower_lim = -Inf;
obs_upper_lim =  Inf;
obsInfo = rlNumericSpec(obs_len, 'LowerLimit', obs_lower_lim, 'UpperLimit',obs_upper_lim);
%obsInfo.Name = 'observation';
%obsInfo.Description = 'instantaneously observed channels';

% Action Specification
act_lower_lim = -Inf;
act_upper_lim =  Inf; % revise limits later for reflection coefficients
actInfo = rlNumericSpec(act_len, 'LowerLimit', act_lower_lim, 'UpperLimit',act_upper_lim);

% write stepfcn
% write resetfcn.m

% Create Environment
MU_MISO_IRS_env = rlFunctionEnv(obsInfo,actInfo,stepfcn,resetfcn);

%% Create Learning Agent
% https://www.mathworks.com/help/reinforcement-learning/ug/ddpg-agents.html
% A DDPG agent consists of two agents: an actor and a critic, cooperating
% together to get a better output action

% ---- Whitening Process for Input Decorrelation --- NOT DONE YET

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


% https://www.mathworks.com/help/reinforcement-learning/ref/rlrepresentationoptions.html
actor_repOpts = rlRepresentationOptions(...
    'Optimizer',used_optmzr,...
    'LearnRate', u_a,...
    'UseDevice',used_device);
% yet to define the decay rate and differentiate between target and actual
% networks

% Create actor agent
ACTOR = rlDeterministicActorRepresentation(actor_net,...
    actor_obsInfo,...
    actor_actInfo,...
    'Observation','channels',...
    'Action','beamformers',...
    actor_repOpts);

% 1-b) Critic Network
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

critic_repOpts = rlRepresentationOptions(...
    'Optimizer',used_optmzr,...
    'LearnRate', u_c,...
    'UseDevice',used_device);
% yet to define the decay rate and differentiate between target and actual
% networks


% Create critic agent
CRITIC = rlQValueRepresentation(critic_net,...
    critic_obsInfo,...
    critic_actInfo,...
    'Observation','states_actions',...
    'Action','Q_approx',...
    critic_repOpts);

%% 3- Specify DDPG options
DDPG_agent_OPTIONS =    rlDDPGAgentOptions('DiscountFactor',gam, ...
    'ExperienceBufferLength',D,...
    'MiniBatchSize', W_exp,...
    'TargetUpdateFrequency', U);

%% 4- Create DDPG agent
% https://www.mathworks.com/help/reinforcement-learning/ref/rlddpgagent.html
DDPG_AGENT = rlDDPGAgent(ACTOR,CRITIC,DDPG_agent_OPTIONS);

%% Train the agent in the environment
% https://www.mathworks.com/help/reinforcement-learning/ug/train-reinforcement-learning-agents.html
% https://www.mathworks.com/help/reinforcement-learning/ref/rl.agent.rlqagent.train.html

% Options
DDPG_train_options = rlTrainingOptions(...
    'MaxEpisodes',N_epis,...
    'MaxStepsPerEpisode',T,...
    'UseParallel', true,...
    'Parallelization', 'async',...
    'Verbose', true,...
    'Plots', 'training-progress');
    
    
% Train DDPG Agent    
trainStats = train(DDPG_AGENT,...           % Agent
                   MU_MISO_IRS_env,...      % Environment
                   DDPG_train_options);     % Training Options


%  % Write Algorithm 1 in paper, then annotate how the MATLAB commands
%  % summarize it 
% ------------------ DONE via TRAIN command -------------------
%     for episodes = 1:N      % loop over episodes 
% ---------------- DONE through RESET function ----------------
%         % Initialize s(1)        
%         for t = 1:T         % move over time instants         
% ---------------- DONE through STEP function ----------------
          % 1- obtain action a from actor network 
%         % 2- observe next state
%         % 3- observe instant reward
%         % 4- store experience in replay memory
%         
%         % Obtain Q-value from critic network
%         % Sample random mini-batches of size _exp of experiences from
%         % experience replay memory \mathcal{M}
%         
%         % Construct critic network loss function
%         % Perform SGD on training and target critic, training actor, to obtain deltas
%         
%         % Update critic then actor networks
%         
%         % Every U steps update the target critic and actor networks
%         
%         % Set input to DNN as s(t+1)
%         end
%     end
