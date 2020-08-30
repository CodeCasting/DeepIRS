%% Code Description and Credits
% This code is to simulate an ML-Driven IRS-aided communication setting,
% where for now, 2 single antenna users communicate with a 2-antenna BS via
% an IRS with M reflecting elements.

% Downlink for now.

% This code is aided by that of the DeepMIMO dataset developed by A.
% Alkhateeb et al.

% It implements the optimization algorithm in [R1].
% [R1] Qingqing Wu, Rui Zhang, "Intelligent Reflecting Surface Enhanced Wireless
% Network via Joint Active and Passive Beamforming", in IEEE Transactions on
% Wireless Communications, Nov. 2019.

% This paper optimizes the active beamforming at the BS and the passive
% reflection at the IRS to minimize the total transmit power under SINR
% QoS constraints.

% The code hence trains and tests a deep neural network (DNN) on the optimized outputs
% and compares the results to the optimized benchmark. The DNN results can
% be obtained in a more timely manner than those obtained via optimization.

%% Simulation Parameters
clear
%close all

profile on

% ---------- Base Station (BS) -----------
% BS is one user with the following row and column indices
Ut_row = 850;               % user Ut row number
Ut_element = 90;            % user Ut col number
% yet to add its MIMO functionality
N_BS = 3;                    % Number of BS antennas
%Pt = 100;                   % Transmit power in dBm

% ----------- Users -----------
N_users= 2;                 % Number of Users
% Users will be randomized from the following region (between the following rows)
% each row contains 181 possible user locations
Ur_rows = [1000 1300];      % user Ur rows
No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of (Ut,Ur) user pairs   %%%%%%%%% GENERALIZE LATER %%%%
% Which row is ommitted? Starting or ending 1? otherwise No_user_pairs
% should be this value +1 --> Last row is not neglected .. ok .. from
% Ur_rows(1) till Ur_rows(2)-1 according to the channel generation code below
RandP_all = randperm(No_user_pairs).'; % Random permutation of the available dataset

all_users = 1:1:N_users;                    % vector of all user indices

int_users_matrix = meshgrid(all_users).';   % indices of interfering users for each user
int_users_matrix(1:N_users+1:N_users^2) = [];
int_users_matrix = reshape(int_users_matrix, N_users-1, N_users).';

% ----------- IRS -----------
% at (one of the dataset BS *locations* - considered passive here of course)
% Also known as LIS: Large Intelligent Surface
% Note: The axes of the antennas match the axes of the ray-tracing scenario
My = 3;                     % number of LIS reflecting elements across the y axis
Mz = 3;                     % number of LIS reflecting elements across the z axis
Mx = 1;                     % number of LIS reflecting elements across the x axis
M = Mx.*My.*Mz;             % Total number of LIS reflecting elements
% M_bar=8;                  % number of active elements, not used here so far
% IRS is assumed at one of the BS locations in the O1 scenario as given by
% params.active_BS.
params.active_BS=3;         % active basestation(/s) in the chosen scenario

% Ray tracing parameters
params.scenario='O1_60';    % DeepMIMO Dataset scenario: http://deepmimo.net/
L =1;                       % number of channel paths (L) .. L=1 returns a narrowband channel with no ISI
kbeams=1;                   % select the top kbeams, get their feedback and find the max actual achievable rate
D_Lambda = 0.5;             % Antenna spacing relative to the wavelength
BW = 100e6;                 % Bandwidth in Hz (ex: 100e6 --> 100 MHz)

% Not used parameters so far since the system is not OFDM
% K_DL=64;                    % number of subcarriers as input to the Deep Learning model
% Validation_Size = 6200;     % Validation dataset Size
% K = 512;                    % number of subcarriers
K=1;                          % No OFDM
K_DL =1;

T = 2e3;                    % number of steps per episode for training
N_epis = 5e1;               % Number of episodes (changed due to DUPLICATE NAME)
sim_len = 1e2;              % Number of generated different multiuser scenarios

sigma_2_dBm = -80; % Noise variance in dBm
sigma_2 = 10^(sigma_2_dBm/10) * 1e-3; % (in Watts)

% SINR target
SINR_target_dB = 0; % Check: changing target SINR should change the transmit power (the cvx_optval objective value)
SINR_target = 10^(SINR_target_dB/10);

% Alternating optimization algorithm (Algorithm 1 in [R1])
% Set error threshold for alternating algorithm termination
eps_iter=1e-1;

%% DeepMIMO Channel Generation (commented for now since it is not available)
% % Using the DeepMIMO Dataset by Alkhateeb et al.
%
% % Select which machine the code is running on
% active_machine = 0;
% if active_machine == 0
%     % Personal machine
%     deepmimo_root_path= 'C:/Khafagy/DeepMIMO'; % Datasets/Large files
%     code_folder = 'C:/Users/Mohammad/Google Drive (mgkhafagy@aucegypt.edu)/MATLAB Codes'; % Code stored on the cloud
% elseif active_machine == 1
%     % Research Lab Windows Workstation
%     deepmimo_root_path= 'D:/Khafagy/DeepMIMO';
%     code_folder = 'C:/Users/Dr. M-Khafagy/Google Drive/MATLAB Codes';
% elseif active_machine == 2
%     % Research Lab Linux Workstation
%     deepmimo_root_path= '/home/khafagy/Storage/DeepMIMO';
%     code_folder = '/home/khafagy/Storage/git/DeepIRS';
% end
% cd(code_folder)
%
% % for dataset retrieval and storage from/to local server (not on the cloud)
%
% %% DeepMIMO Dataset Generation (CALLING DeepMIMO_generator)
% % DeepMIMO_generator calls read_raytracing then construct_DeepMIMO_channel
% % These code files are created by Alkhateeb et al.
% disp('===============GENERATING DEEPMIMO DATASET===================');
% %disp('-------------------------------------------------------------');
% %disp([' Calculating for K_DL = ' num2str(K_DL)]);
% % ------  Inputs to the DeepMIMO dataset generation code ------------ %
% % Note: The axes of the antennas match the axes of the ray-tracing scenario
% params.num_ant_x= Mx;             % Number of the UPA antenna array on the x-axis
% params.num_ant_y= My;             % Number of the UPA antenna array on the y-axis
% params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
% params.ant_spacing=D_Lambda;      % ratio of the wavelnegth; for half wavelength enter .5
% params.bandwidth= BW*1e-9;        % The bandiwdth in GHz
% params.num_OFDM= K;               % Number of OFDM subcarriers
% params.OFDM_sampling_factor=1;    % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
% params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
% params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
% params.saveDataset=0;
% disp([' Calculating for L = ' num2str(params.num_paths)]);
%
% %% BS-IRS Channels
% disp('==========Generating Transmit BS-IRS Full Channel============');
% % ------------------ DeepMIMO "Ut" Dataset Generation ----------------- %
% params.active_user_first=Ut_row;
% params.active_user_last=Ut_row;                                 % Only one active user (but where is Ut_element to fully specify the user??) -- see below
% DeepMIMO_dataset=DeepMIMO_generator(params,deepmimo_root_path); % Generator function generates data for entire rows
% %Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);     % Selecting element of interest here
% Ht = DeepMIMO_dataset{1}.user{Ut_element}.channel;              % Selecting element of interest here
%
% clear DeepMIMO_dataset
%
% % ----------- Add BS MIMO Functionality here -------
% % Remember to later randomize the transmitter as well, so that the neural
% % network is not a function of a fixed BS-IRS channel
%
% % Adjust size for now (simply replicate), then fix the MIMO functionality later
% %Ht = repmat(Ht,1, N_BS);
% Ht = 1e-4/sqrt(2)*(randn(M, N_BS)+1i*randn(M, N_BS));
%
% %% IRS - Receiver Channels
% disp('===========Generating IRS-Receiver Full Channels=============');
% % ------------------ DeepMIMO "Ur" Dataset Generation -----------------%
% %initialization
% Ur_rows_step = 300; % access the dataset 100 rows at a time
% Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2);
% Delta_H_max = single(0);
% for pp = 1:1:numel(Ur_rows_grid)-1          % loop for Normalizing H
%     clear DeepMIMO_dataset
%     params.active_user_first=Ur_rows_grid(pp);
%     params.active_user_last=Ur_rows_grid(pp+1)-1;
%     disp(['=== User Row Batch ' num2str(pp) ' out of ' num2str(numel(Ur_rows_grid)-1) ', each holding ' num2str(Ur_rows_step) ' rows ====='])
%     [DeepMIMO_dataset,params]=DeepMIMO_generator(params,deepmimo_root_path);
%     for u=1:params.num_user                 % seems to be hard-coded as rows*181 already
%         Hr = single(conj(DeepMIMO_dataset{1}.user{u}.channel));    % conjugated since it is now downlink
%         Delta_H = max(max(abs(Ht.*Hr)));
%         if Delta_H >= Delta_H_max
%             Delta_H_max = single(Delta_H);  % storing the maximum absolute value of the end-to-end product channel for later normalization
%         end
%     end
% end
% clear Delta_H

%% Pregenerate Training Channels
H_mat.Hd_mat = 1e-4/sqrt(2)*(randn(N_BS, N_users, N_epis)+1i*randn(N_BS, N_users, N_epis));
H_mat.Hr_mat = 1e-2/sqrt(2)*(randn(M, N_users, N_epis)+1i*randn(M, N_users, N_epis));
H_mat.Ht_mat = 1e-2/sqrt(2)*(randn(M, N_BS, N_epis)+1i*randn(M, N_BS, N_epis));

%% Create and Train DDPG AGENT
drl_IRS

%% Generate different channels for testing

Hd_mat = 1e-4/sqrt(2)*(randn(N_BS, N_users, sim_len)+1i*randn(N_BS, N_users, sim_len));
Hr_mat = 1e-2/sqrt(2)*(randn(M, N_users, sim_len)+1i*randn(M, N_users, sim_len));
Ht_mat = 1e-2/sqrt(2)*(randn(M, N_BS, sim_len)+1i*randn(M, N_BS, sim_len));


%% Loop over different user permutations and store optimized solutions
tic
%clear H W

ML_dataset{sim_len} = {}; % Store channels, locations, and solutions
user_loc{N_users} = {};

% Fix seed
rng(1);

myCluster = parcluster();
if isempty(gcp)
    myPool = parpool(myCluster);
end

disp('Looping over different multi-user patterns and generating optimized matrices')
parfor sim_index = 1:sim_len
    disp(['=== User pattern ' num2str(sim_index) ' out of ' num2str(sim_len) ' ====='])
    
    % Select N_users random user indices
    %     clear Hr
    %     Hr{N_users} = [];
    %     users = randperm(params.num_user, N_users);
    %     for user_ind = 1:N_users
    %         Hr{user_ind} = DeepMIMO_dataset{1}.user{users(user_ind)}.channel;
    %         %user_loc{user_ind} = DeepMIMO_dataset{1}.user{users(user_ind)}.loc;
    %     end
    %     Hr = [Hr{:}];
    
    frac_error=1e10;    % Initialize fractional error
    obj_last = 1e3; % Initialize last objective value to a large number
    
    % Implement Optimization algorithm here
    
    % Let the direct channel be denoted by Hsd (source to destination)
    %Hd = zeros(N_BS, N_users);
    Hd = Hd_mat(:,:,sim_index);
    Hr = Hr_mat(:,:,sim_index);
    Ht = Ht_mat(:,:,sim_index);
    
    ML_dataset{sim_index}.Ht = Ht; % Store transmit (1st hop) channel
    ML_dataset{sim_index}.Hd = Hd;  % Store direct channel
    ML_dataset{sim_index}.Hr = Hr;  % Store receive (2nd hop) channel
    %ML_dataset{sim_index}.user_loc = [user_loc{:}]; % Store user_locations
    
    %% Alternating Optimization
    %disp('Running alternating optimization algorithm')
    r=1;            % iteration index
    % Initialize reflection matrix theta
    beta_vec = ones(M,1);               % Fixed to 1 for now as in the paper
    theta_vec = 2*pi*rand(M,1);          % Uniformly randomized from 0 to 2*pi
    theta_mat= diag(beta_vec.*exp(1i*theta_vec));
    
    H = Ht'*(theta_mat')*Hr + Hd;
    
    % Check rank criterion for feasbility of the initial theta choice
    while ~(rank(H) == N_users) % if infeasible choice, randomize and check again
        %disp('infeasible initial choice of theta, .. reselecting ..')
        theta_vec = 2*pi*rand(M,1);           % Uniformly randomized from 0 to 2*pi
        theta_mat= diag(beta_vec.*exp(1i*theta_vec));
        H = Ht'*(theta_mat')*Hr + Hd;
    end
    
    cvx_status = 'nothing'; % initialize
    
    while (frac_error > eps_iter)  && ~contains(cvx_status,'Infeasible','IgnoreCase',true)
        %     if mod(r,1e2)==0
        %         %disp(['Iteration r =' num2str(r)])
        %     end
        
        H = Ht'*(theta_mat')*Hr + Hd;
        
        % ==== Optimize W while fixing theta ==== BS Transmit Beamforming
        %disp('Active Beamformer Design')
        
        [W, tau, INTERF, cvx_status, cvx_optval] = iter_opt_prob_1(H,sigma_2,SINR_target,int_users_matrix);
        
        if  cvx_optval==Inf
            %disp('Infeasible .. passing this iteration')
            continue
        end
        %disp(['CVX Status: ' cvx_status ', CVX_optval = ' num2str(10*log10(cvx_optval*1000)) ' dBm'])
        %disp(['CVX Status: ' cvx_status ', CVX_optval = ' num2str(10*log10(trace(W'*W)*1000)) ' dBm'])
        
        frac_error = abs(obj_last - cvx_optval)/obj_last *100;
        obj_last = cvx_optval;
        
        achieved_SINR = zeros(1,N_users);
        % Actual achieved SINR
        for k = all_users
            achieved_SINR(k) = (norm((H(:,k)')*W(:,k)))^2/(norm(INTERF(:,k)))^2;
        end
        
        
        % ==== Optimize theta while fixing W ==== IRS Reflection Matrix
        % (P4') in paper
        %disp('Passive Beamformer Design')
        
        [V, a_aux, a, b, R, desired, interference, SINR_CONSTR, cvx_status, cvx_optval] = iter_opt_prob_2(W, Ht,Hr,Hd,sigma_2,SINR_target,int_users_matrix);
        
        %disp(['CVX Status: ' cvx_status])
        
        if ~contains(cvx_status,'Infeasible','IgnoreCase',true)
            %disp('Running Gaussian Randomization')
            [U,D] = eig(full(V));                         % Eigenvalue Decomposition
            if rank(full(V)) == 1
                v_bar = U*sqrt(D);
                theta_vec = angle(v_bar(1:M)/v_bar(M+1));
                v = exp(-1i*theta_vec);
                theta_mat = diag(v);
                
            else             % Apply Gaussian Randomization
                
                num_rands = 1e3;                        % number of randomizations
                
                % Generate Gaussian random vector ~ CN(0, I)
                %gpudev = gpuDevice();
                %reset(gpudev);
                r_vec_matrix = (1/sqrt(2))*((mvnrnd(zeros(M+1,1),eye(M+1),num_rands) + 1i * mvnrnd(zeros(M+1,1),eye(M+1), num_rands)).'); %gpuArray()
                v_bar_matrix = U*sqrt(D)*r_vec_matrix;
                
                best_index = 0;
                best_value = -1e8;
                %v_bar_matrix = exp(1i*2*pi*rand(M+1,num_rands));
                
                for randmzn_index = 1:num_rands
                    v_bar_vec = v_bar_matrix(:,randmzn_index);
                    V_rand = v_bar_vec*(v_bar_vec');
                    
                    [~, ~, constr_value] = sinr_CONSTRAINT(V_rand, b, R, SINR_target, sigma_2, all_users, int_users_matrix);
                    
                    % Check feasibility and best value
                    feasibility_check = prod( constr_value >=  0 );
                    better_value_check = (sum(constr_value) > best_value);
                    if  feasibility_check && better_value_check
                        best_index = randmzn_index;
                        best_value = sum(constr_value);
                    end
                end
                
                if best_index ~= 0
                    % select best v_bar that maximizes SINR_CONSTR
                    v_bar = v_bar_matrix(:,best_index);
                    theta_vec = angle(v_bar(1:M)/v_bar(M+1));
                    v = exp(-1i*theta_vec);
                    theta_mat = diag(v);
                else
                    cvx_status = 'Infeasible';
                end
                
                %disp(['CVX Status after randomization: ' cvx_status])
            end
        end
        
        %     % Increment iteration index
        r = r+1;
    end
    
    %%
    ML_dataset{sim_index}.W_OPT = W;  % Store Transmit Beamformer
    ML_dataset{sim_index}.theta_OPT = -1i*log(diag(theta_mat));  % Store Reflection Coefficients
    ML_dataset{sim_index}.iterations = r-1;
    
    % ----------- end iterative algorithm ------------------
    
    % Get DRL Solution
    chan_obs =  [  real(Ht(:)); imag(Ht(:));
        real(Hr(:)); imag(Hr(:));
        real(Hd(:)); imag(Hd(:))];
    switch chan_state_design
        case 1
            obs = [chan_obs; past_action_default];
        case 2
            obs = chan_obs;
    end
    Action = DDPG_AGENT.getAction(obs);
    ML_dataset{sim_index}.W_DRL = reshape(Action(1:N_BS*N_users)+ 1i*Action(N_BS*N_users+1:2*N_BS*N_users), N_BS, N_users);
    ML_dataset{sim_index}.theta_DRL = Action(2*N_BS*N_users+1:2*N_BS*N_users+M);
    
end

delete(gcp('nocreate'))

% save([deepmimo_root_path '/saved_datasets.mat'], 'ML_dataset')
disp(['Elapsed time = ' num2str(toc/60) ' minutes.'])

%% Build Neural Network here

% % For regression neural network, we can directly use newgrnn
% tic
% % Prepare INPUT and OUTPUT matrices
% INPUT = zeros(sim_len,2*(M * N_users + M * N_BS + N_BS* N_users)); % The 3 vectorized channel matrices
% OUTPUT = zeros(sim_len, 2*(M + N_BS* N_users)); % Vectorized beamformers
% iterations = zeros(sim_len,1);
% % Generalized Regression Neural Networks in MATLAB
% for loop_index = 1:sim_len
%     TEMP = ML_dataset{loop_index};
%     INPUT(loop_index,:)  = [real(TEMP.Ht(:)); imag(TEMP.Ht(:));
%         real(TEMP.Hr(:)); imag(TEMP.Hr(:));
%         real(TEMP.Hd(:)); imag(TEMP.Hd(:))].';
%     OUTPUT(loop_index,:) = [real(TEMP.W(:)); imag(TEMP.W(:));
%         real(TEMP.theta(:)); imag(TEMP.theta(:))].';
%     iterations(loop_index) = TEMP.iterations;
% end
%
% net = newgrnn(INPUT.',OUTPUT.');
% y = net(INPUT.').';
%
% toc

% Training_Size=[2  1e4*(1:.4:3)];        % Training Dataset Size vector
% % Should be made a function of sim_len: the size of our stored optimized
% % data, which will be split into training, testing, and validation
%
% Validation_Size = 6200;                 % Validation dataset Size
% miniBatchSize  = 500;                   % Size of the minibatch for the Deep Learning

% disp('======================DL BEAMFORMING=========================');
% % ------------------ Training and Testing Datasets -----------------%
% for dd=1:1:numel(Training_Size)
%     disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
%     Training_Ind   = RandP_all(1:Training_Size(dd));
%
%     XTrain = single(DL_input_reshaped(:,1,1,Training_Ind));
%     YTrain = single(DL_output_reshaped(1,1,:,Training_Ind));
%     XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
%     YValidation = single(DL_output_reshaped(1,1,:,Validation_Ind));
%     YValidation_un = single(DL_output_reshaped_un);
%
%     % ------------------ DL Model definition -----------------%
%     % Layers
%     layers = [
%         % INPUT Layer
%         imageInputLayer([size(XTrain,1),1,1],'Name','input')
%
%         % Fully Connected Layer 1 with Dropout
%         fullyConnectedLayer(size(YTrain,3),'Name','Fully1')
%         reluLayer('Name','relu1')
%         dropoutLayer(0.5,'Name','dropout1')
%
%         % Fully Connected Layer 2 with Dropout
%         fullyConnectedLayer(4*size(YTrain,3),'Name','Fully2')
%         reluLayer('Name','relu2')
%         dropoutLayer(0.5,'Name','dropout2')
%
%         % Fully Connected Layer 3 with Dropout
%         fullyConnectedLayer(4*size(YTrain,3),'Name','Fully3')
%         reluLayer('Name','relu3')
%         dropoutLayer(0.5,'Name','dropout3')
%
%         % OUTPUT Layer
%         fullyConnectedLayer(size(YTrain,3),'Name','Fully4')
%         regressionLayer('Name','outReg')];
%
%     if Training_Size(dd) < miniBatchSize
%         validationFrequency = Training_Size(dd);
%     else
%         validationFrequency = floor(Training_Size(dd)/miniBatchSize);
%     end
%     VerboseFrequency = validationFrequency;
%
%     % Options
%     options = trainingOptions('sgdm', ...
%         'MiniBatchSize',miniBatchSize, ...
%         'MaxEpochs',20, ...
%         'InitialLearnRate',1e-1, ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropFactor',0.5, ...
%         'LearnRateDropPeriod',3, ...
%         'L2Regularization',1e-4,...
%         'Shuffle','every-epoch', ...
%         'ValidationData',{XValidation,YValidation}, ...
%         'ValidationFrequency',validationFrequency, ...
%         'Plots','none', ... % 'training-progress'
%         'Verbose',0, ...    % 1
%         'ExecutionEnvironment', 'cpu', ...
%         'VerboseFrequency',VerboseFrequency);
%
%     % ------------- DL Model Training and Prediction -----------------%
%     [~,Indmax_OPT]= max(YValidation,[],3);
%     Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
%     MaxR_OPT = single(zeros(numel(Indmax_OPT),1));
%
%     [trainedNet,traininfo]  = trainNetwork(XTrain,YTrain,layers,options);
%
%     YPredicted = predict(trainedNet,XValidation);
%
%     % --------------------- Achievable Rate --------------------------% <--- change
%     [~,Indmax_DL] = maxk(YPredicted,kbeams,2);
%     MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rates
%     for b=1:size(Indmax_DL,1)
%         MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b)));
%         MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));
%     end
%
%     % shall be removed
%     Rate_OPT(dd) = mean(MaxR_OPT);
%     Rate_DL(dd) = mean(MaxR_DL);
%     LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);
%
%     clear trainedNet traininfo YPredicted
%     clear layers options
% end

%% Plot Figures
close all


% Plot Actor and Critic Neural Networks
figure(1)
subplot(1,2,1)
plot(actor_net)
title('Actor Network')
subplot(1,2,2)
plot(critic_net)
title('Critic Network')

% Prelim calculations for figures

SINR_DRL = zeros(N_users,sim_len);
SINR_OPT = zeros(N_users,sim_len);
transmit_pow_DRL = zeros(sim_len,1);
transmit_pow_OPT = zeros(sim_len,1);

for sim_index = 1:sim_len
    Ht = ML_dataset{sim_index}.Ht;
    Hd = ML_dataset{sim_index}.Hd;
    Hr = ML_dataset{sim_index}.Hr;
    % DRL Solutions
    % Extract BS beamformer and IRS reflection matrix from taken action
    W_DRL = ML_dataset{sim_index}.W_DRL;
    theta_vec_DRL = ML_dataset{sim_index}.theta_DRL;
    theta_mat_DRL = diag(exp(1i*theta_vec_DRL));
    H_DRL = Ht'*(theta_mat_DRL')*Hr + Hd;
    
    W_OPT = ML_dataset{sim_index}.W_OPT;
    theta_vec_OPT = ML_dataset{sim_index}.theta_OPT;
    theta_mat_OPT = diag(exp(1i*theta_vec_OPT));
    H_OPT = Ht'*(theta_mat_OPT')*Hr + Hd;
    
    transmit_pow_DRL(sim_index) = sum([diag(real(W_DRL)'*real(W_DRL)); diag(imag(W_DRL)'*imag(W_DRL))]);
    transmit_pow_OPT(sim_index) = sum([diag(real(W_OPT)'*real(W_OPT)); diag(imag(W_OPT)'*imag(W_OPT))]);
    
    for  user_ind = 1 : N_users
        int_users = int_users_matrix(user_ind,:); % interfering user indices
        % DRL
        desired_DRL = W_DRL(:,user_ind)'*H_DRL(:,user_ind);
        interf_DRL = [W_DRL(:,int_users)'*H_DRL(:,user_ind); sqrt(sigma_2)];
        SINR_DRL(user_ind,sim_index) = norm(desired_DRL,2)^2/norm(interf_DRL,2)^2;
        
        % OPT
        desired_OPT = W_OPT(:,user_ind)'*H_OPT(:,user_ind);
        interf_OPT = [W_OPT(:,int_users)'*H_OPT(:,user_ind); sqrt(sigma_2)];
        SINR_OPT(user_ind,sim_index) = norm(desired_OPT,2)^2/norm(interf_OPT,2)^2;
    end
end

min_SINR_DRL = min(SINR_DRL,[],1);
min_SINR_OPT = min(SINR_OPT,[],1);

% Min SINR compared to threshold vs. simulation index
figure(2)
plot(1:sim_len, SINR_target*ones(1,sim_len), 'r-')
title('Minimum SINR achieved for all users')
xlabel('Simulation Index')
ylabel('Minimum SINR')
hold on
plot(1:sim_len, min_SINR_DRL, 'b-')
plot(1:sim_len, min_SINR_OPT, 'k-')

legend('Threshold', 'DRL', 'Alternating Opt.')

% Power Consumption
figure(3)
title('BS Transmit Power')
xlabel('Simulation Index')
ylabel('Transmit Power (W)')
hold on
plot(1:sim_len, transmit_pow_DRL, 'b-')
plot(1:sim_len, transmit_pow_OPT, 'k-')

legend('DRL', 'Alternating Opt.')

% Time/Complexity
