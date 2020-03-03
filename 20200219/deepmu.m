%% Code Description and Credits
% This code is to simulate an ML-Driven IRS-aided communication setting,
% where for now, 2 single antenna users communicate with a 2-antenna BS via
% an IRS with M reflecting elements.

% Downlink for now.

% This code is aided by that of the DeepMIMO dataset developed by A.
% Alkhateeb et al.

% It implements the optimization algorithms of the paper below
% [R1] Qingqing Wu, Rui Zhang, "Intelligent Reflecting Surface Enhanced Wireless
% Network via Joint Active and Passive Beamforming", in IEEE Transactions on
% Wireless Communications, Nov. 2019.

% This paper optimizes the active beamforming at the BS and the passive
% reflection at the IRS to minimize the total transmit power under SINR
% QoS constraints.

% It hence trains and tests a deep neural network (DNN) on the optimized outputs
% and compares the results to the optimized benchmark. The DNN results can
% be obtained in a more timely manner than those obtained via optimization.

%% Simulation Parameters
clear
close all

% ---------- Base Station (BS) -----------
% BS is one user with the following row and column indices
Ut_row = 850;               % user Ut row number
Ut_element = 90;            % user Ut col number
% yet to add its MIMO functionality
N_BS = 5;                   % Number of BS antennas
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
My = 6;                     % number of LIS reflecting elements across the y axis
Mz = 6;                     % number of LIS reflecting elements across the z axis
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

sim_len = 1e2;              % Number of generated different multiuser scenarios

%% Channel Generation
% Using the DeepMIMO Dataset by Alkhateeb et al.

% Select which much the code is running on
personal = 1;
if personal == 1
    % Personal machine
    deepmimo_root_path= 'C:/Khafagy/DeepMIMO';
    code_folder = 'C:/Users/Mohammad/Google Drive (mgkhafagy@aucegypt.edu)/MATLAB Codes';
elseif personal == 0
    % Research Lab Workstation
    deepmimo_root_path= 'D:/Khafagy/DeepMIMO';
    code_folder = 'C:/Users/Dr. M-Khafagy/Google Drive/MATLAB Codes';
end
cd(code_folder)

% for dataset retrieval and storage from/to local server (not on the cloud)

% %% Beamforming Codebook (CALLING UPA_codebook_generator)
% disp('=============GENERATING REFLECTION MATRIX CODEBOOK===========');
% % BF codebook parameters
% over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
% over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
% over_sampling_z=1;            % The beamsteering oversampling factor in the z direction
%
% % Generating the BF codebook
% % [BF_codebook]=sqrt(Mx*My*Mz)*...
% %     UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
% % codebook_size=size(BF_codebook,2);

%% DeepMIMO Dataset Generation (CALLING DeepMIMO_generator)
% DeepMIMO_generator calls read_raytracing then construct_DeepMIMO_channel
% These code files are created by Alkhateeb et al.
disp('===============GENERATING DEEPMIMO DATASET===================');
%disp('-------------------------------------------------------------');
%disp([' Calculating for K_DL = ' num2str(K_DL)]);
% ------  Inputs to the DeepMIMO dataset generation code ------------ %
% Note: The axes of the antennas match the axes of the ray-tracing scenario
params.num_ant_x= Mx;             % Number of the UPA antenna array on the x-axis
params.num_ant_y= My;             % Number of the UPA antenna array on the y-axis
params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
params.ant_spacing=D_Lambda;      % ratio of the wavelnegth; for half wavelength enter .5
params.bandwidth= BW*1e-9;        % The bandiwdth in GHz
params.num_OFDM= K;               % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;    % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
params.saveDataset=0;
disp([' Calculating for L = ' num2str(params.num_paths)]);

%% BS-IRS Channels
disp('==========Generating Transmit BS-IRS Full Channel============');
% ------------------ DeepMIMO "Ut" Dataset Generation ----------------- %
params.active_user_first=Ut_row;
params.active_user_last=Ut_row;                                 % Only one active user (but where is Ut_element to fully specify the user??) -- see below
DeepMIMO_dataset=DeepMIMO_generator(params,deepmimo_root_path); % Generator function generates data for entire rows
%Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);     % Selecting element of interest here
Ht = DeepMIMO_dataset{1}.user{Ut_element}.channel;              % Selecting element of interest here

clear DeepMIMO_dataset

% ----------- Add BS MIMO Functionality here -------
% Remember to later randomize the transmitter as well, so that the neural
% network is not a function of a fixed BS-IRS channel

% Adjust size for now (simply replicate), then fix the MIMO functionality later
%Ht = repmat(Ht,1, N_BS);
Ht = 1e-2/sqrt(2)*(randn(M, N_BS)+1i*randn(M, N_BS));

%% IRS - Receiver Channels
disp('===========Generating IRS-Receiver Full Channels=============');
% ------------------ DeepMIMO "Ur" Dataset Generation -----------------%
%initialization
Ur_rows_step = 300; % access the dataset 100 rows at a time
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2);
Delta_H_max = single(0);
for pp = 1:1:numel(Ur_rows_grid)-1          % loop for Normalizing H
    clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    disp(['=== User Row Batch ' num2str(pp) ' out of ' num2str(numel(Ur_rows_grid)-1) ', each holding ' num2str(Ur_rows_step) ' rows ====='])
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params,deepmimo_root_path);
    for u=1:params.num_user                 % seems to be hard-coded as rows*181 already
        Hr = single(conj(DeepMIMO_dataset{1}.user{u}.channel));    % conjugated since it is now downlink
        Delta_H = max(max(abs(Ht.*Hr)));
        if Delta_H >= Delta_H_max
            Delta_H_max = single(Delta_H);  % storing the maximum absolute value of the end-to-end product channel for later normalization
        end
    end
end
clear Delta_H

%% Loop over different user permutations and store optimized solutions
ML_dataset{sim_len} = {}; % Store channels, locations, and solutions
user_loc{N_users} = {};

% Fix seed
rng(1);

disp('Looping over different multi-user patterns and generating optimized matrices')
for sim_index = 1:sim_len
    disp(['=== User pattern ' num2str(sim_index) ' out of ' num2str(sim_len) ' ====='])
    ML_dataset{sim_index}.Ht = Ht; % Store transmit (1st hop) channel
    
    % Select N_users random user indices
    clear Hr
    Hr{N_users} = [];
    user_loc{N_users} = {};
    users = randperm(params.num_user, N_users);
    for user_ind = 1:N_users
        Hr{user_ind} = DeepMIMO_dataset{1}.user{users(user_ind)}.channel;
        user_loc{user_ind} = DeepMIMO_dataset{1}.user{users(user_ind)}.loc;
    end
    Hr = [Hr{:}];
    
    ML_dataset{sim_index}.Hr = Hr;  % Store receive (2nd hop) channel
    ML_dataset{sim_index}.user_loc = [user_loc{:}]; % Store user_locations
    
    % Implement Optimization algorithm here
    
    % Let the direct channel be denoted by Hsd (source to destination)
    %Hd = zeros(N_BS, N_users);
    Hd = 1e-10/sqrt(2)*(randn(N_BS, N_users)+1i*randn(N_BS, N_users));
    
    ML_dataset{sim_index}.Hd = Hd;  % Store direct channel
    
    sigma_2_dBm = -80; % Noise variance in dBm
    sigma_2 = 10^(sigma_2_dBm/10) * 1e-3; % (in Watts)
    
    % SINR target
    SINR_target_dB = 15; % Check: changing target SINR should change the transmit power (the cvx_optval objective value)
    SINR_target = 10^(SINR_target_dB/10);
    
    % Alternating optimization algorithm (Algorithm 1 in [R1])
    % Set error threshold for alternating algorithm termination
    eps_iter=1e0;
    frac_error=1e10;    % Initialize fractional error
    obj_last = 1e3; % Initialize last objective value to a large number
    
    disp('Running alternating optimization algorithm')
    r=1;            % iteration index
    % Initialize reflection matrix theta
    beta_vec = ones(M,1);               % Fixed to 1 for now as in the paper
    theta_vec = ones(M,1); % 2*pi*rand(M,1);          % Uniformly randomized from 0 to 2*pi
    theta_mat= diag(beta_vec.*exp(1i*theta_vec));
    
    % Check rank criterion for feasbility of the initial theta choice
    while ~(rank(Ht'*theta_mat*Hr + Hd) == N_users) % if infeasible choice, randomize and check again
        disp('infeasible initial choice of theta, .. reselecting ..')
        theta_vec = 2*pi*rand(M,1);           % Uniformly randomized from 0 to 2*pi
        theta_mat= diag(beta_vec.*exp(1i*theta_vec));
    end
    
    cvx_status = 'nothing'; % initialize
    
    while (frac_error > eps_iter)  && ~contains(cvx_status,'Infeasible','IgnoreCase',true)
        if mod(r,1e2)==0
            disp(['Iteration r =' num2str(r)])
        end
        
        % ==== Optimize W while fixing theta ==== BS Transmit Beamforming
        disp('Active Beamformer Design')
        
        cvx_clear
        clear W tau INTERFERENCE obj_fn1 cvx_optval
        
        cvx_begin
        cvx_quiet(true)
        cvx_solver  SDPT3 %SDPT3  %Mosek     % Choose the underlying solver
        cvx_precision best        % Change the cvx numerical precision
        
        % Define your optimization variables here
        variable W(N_BS,N_users) complex; % add the word binary for binary constraints - see CVX documentation for more options
        variable tau nonnegative; % Auxiliary variable
        expressions INTERFERENCE(N_users,N_users);
        
        for k = all_users
            int_users = int_users_matrix(k,:); % interfering users
            INTERFERENCE(:,k) = [ ...
                W(:,int_users)'*(Ht'*(theta_mat')*(Hr(:,k)) + (Hd(:,k)));
                sqrt(sigma_2)];
        end
        
        % Write the optimization problem
        minimize( tau^2 );
        subject to
        for k = all_users
            {INTERFERENCE(:,k), sqrt(1/SINR_target)*real(((Hr(:,k)'*theta_mat*Ht + Hd(:,k)')*W(:,k)))} == complex_lorentz(N_users); % SINR CONSTRAINT
        end
        {W(:), tau} == complex_lorentz(N_BS * N_users); % POWER CONSTRAINT
        cvx_end
        disp(['CVX Status: ' cvx_status ', CVX_optval = ' num2str(10*log10(cvx_optval*1000)) ' dBm'])
        
        frac_error = abs(obj_last - cvx_optval)/obj_last *100;
        obj_last = cvx_optval;
        
        achieved_SINR = zeros(1,N_users);
        % Actual achieved SINR 
        for k = all_users
        achieved_SINR(k) = (norm((Hr(:,k)'*theta_mat*Ht+ Hd(:,k)')*W(:,k)))^2/(norm(INTERFERENCE(:,k)))^2;
        end
%         achieved_SINR
%         trace(W*(W'))
%         cvx_optval
%         10*log10(trace(W*(W'))*1000)
        
        % ==== Optimize theta while fixing W ==== IRS Reflection Matrix
        % (P4') in paper
        disp('Passive Beamformer Design')
        
        % Define a, b and R
        a = cell(N_users,N_users);
        b = cell(N_users,N_users);
        R = cell(N_users,N_users);
        for k = all_users                               % looping over all users
            int_users = int_users_matrix(k,:);          % interfering users
            a{k,k}= diag(Hr(:,k)')*Ht*W(:,k);
            b{k,k}= Hd(:,k)'*W(:,k);
            R{k,k}= [ a{k,k}* (a{k,k}')  a{k,k}* (b{k,k}') ;  a{k,k}'* b{k,k} 0];
            for m = int_users
                a{k,m}= diag(Hr(:,k)')*Ht*W(:,m);
                b{k,m}= Hd(:,k)'*W(:,m);
                R{k,m}= [ a{k,m}* (a{k,m}')  a{k,m}* (b{k,m}') ;  a{k,m}'* b{k,m} 0];
            end
        end
        
        cvx_begin sdp
        cvx_quiet(true)
        cvx_solver SDPT3 %SeDuMi %SDPT3  %Mosek     % Choose the underlying solver
        cvx_precision best        % Change the cvx numerical precision
                
        variable V(M+1,M+1) complex semidefinite;
        variable a_aux(1,N_users) nonnegative;   % Auxiliary variables for max sum
        %variable a_aux nonnegative;       % Auxiliary variables for max min
        expressions  SINR_CONSTR(N_users) desired(N_users) interference(N_users);
        
        % Define the expressions desired, interference, and SINR_CONSTR in terms of the optimization variables 
        %sinr_fun_handle = @sinr_CONSTRAINT;
                
        %[desired, interference, SINR_CONSTR] = sinr_fun_handle(V, b, R, SINR_target, sigma_2, all_users, int_users_matrix);
        for k = all_users                               % looping over all users
            int_users = int_users_matrix(k,:);          % interfering users
            desired(k) = trace(real(R{k,k}*V)) + square_abs(b{k,k});
            interference(k) = 0;
            for m = int_users
                interference(k) = interference(k) + trace(real(R{k,m}*V)) + square_abs(b{k,m});
            end
            SINR_CONSTR(k) =    desired(k)  -  a_aux(k) - SINR_target * (interference(k) + sigma_2);
        end

        %all_elements = 1:M+1;
        
        % Write the optimization problem
        maximize( sum(a_aux) );
        subject to
        %SINR_CONSTR == nonnegative(N_users)
        for k = 1:N_users
            desired(k)   >= a_aux(k) +  SINR_target * (interference(k) + sigma_2);
            %sqrt(SINR_CONSTR(k)) >= 0
        end
        diag(V) == ones(M+1,1);
        % Other 2 constraints are already in the definitions of opt variables
        %obj_fn2 >= 0; % Dummy constraint to check/prevent the resulting -ve cvx_optval
        cvx_end
        
        disp(['CVX Status: ' cvx_status])
        
        if ~contains(cvx_status,'Infeasible','IgnoreCase',true)
            disp('Running Gaussian Randomization')
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
                    feasibility_check = prod( constr_value >=  a_aux );
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
                
                disp(['CVX Status after randomization: ' cvx_status])
            end
        end
        
        % Increment iteration index
        r = r+1;
    end
    
    ML_dataset{sim_index}.W = W;  % Store Transmit Beamformer
    ML_dataset{sim_index}.theta = diag(theta_mat);  % Store Reflection Matrix Diagonal
    ML_dataset{sim_index}.iterations = r-1;
    
    % ----------- end iterative algorithm ------------------
    
end
save([deepmimo_root_path '/saved_datasets.mat'], 'ML_dataset')

%% Build Neural Network here

% For regression neural network, we can directly use newgr

% Prepare INPUT and OUTPUT matrices
INPUT = zeros(sim_len,2*(M * N_users + M * N_BS + N_BS* N_users)); % The 3 vectorized channel matrices
OUTPUT = zeros(sim_len, 2*(M^2 + N_BS* N_users)); % Vectorized beamformers
iterations = zeros(sim_len,1);
% Generalized Regression Neural Networks in MATLAB
for loop_index = 1:sim_len
    TEMP = ML_dataset{loop_index};
    INPUT(loop_index,:)  = [real(TEMP.Ht(:)); imag(TEMP.Ht(:));
        real(TEMP.Hr(:)); imag(TEMP.Hr(:));
        real(TEMP.Hd(:)); imag(TEMP.Hd(:))].';
    OUTPUT(loop_index,:) = [real(TEMP.W(:)); imag(TEMP.W(:));
        real(TEMP.theta(:)); imag(TEMP.theta(:))].';
    iterations(loop_index) = TEMP.iterations;
end

net = newgrnn(INPUT.',OUTPUT.');
y = net(INPUT.').';

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
% Power Consumption

% Time/Complexity


