% Code Description
% ----------------
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html
% https://www.mathworks.com/help/reinforcement-learning/ug/define-reward-signals.html

% Input is the action taken by the actor agent in DDPG
% This step function calculates the new state/observation and reward
% due to taken (input) action wich the agent has just taken.

function [new_observation,Reward,IsDone,LoggedSignals] = stepfcn_power(action, LoggedSignals)

% -------------- Extract Logged Signals ------------------
sigma_2 = LoggedSignals.sigma_2; % noise variance
% Extract current channel from logged signals
Ht = LoggedSignals.new_chan_obs.Ht;
Hr = LoggedSignals.new_chan_obs.Hr;
Hd = LoggedSignals.new_chan_obs.Hd;
% Numbers of Users, IRS reflecting elements, and BS Antennas
N_users = size(Hr,2); M = size(Ht,1); N_BS = size(Ht,2);

% Extract BS beamformer from taken action
W = reshape(action(1:N_BS*N_users)+ 1i*action(N_BS*N_users+1:2*N_BS*N_users), N_BS, N_users);
% Extract IRS reflection vector from taken action
theta_vec = action(2*N_BS*N_users+1:2*N_BS*N_users+M);
theta_mat = diag(exp(1i*theta_vec));

% Extract past action from Logged signals
% past_action = LoggedSignals.Action;
% Calculate transmit power for each user (stacking real and imag powers)
transmit_pow = [diag(real(W)'*real(W)); diag(imag(W)'*imag(W))];
% Calculate received power for each user (also stacking real and imag)
H_W = W'*(Ht'*(theta_mat')*Hr + Hd);
H_real_imag_vec = [real(H_W(:)); imag(H_W(:))];
receive_pow = H_real_imag_vec.^2; 

% Channel observation
chan_obs = [real(Ht(:)); imag(Ht(:)); real(Hr(:)); imag(Hr(:)); real(Hd(:)); imag(Hd(:))];
%new_observation = [transmit_pow; receive_pow; chan_obs; action];
new_observation = chan_obs;
            
int_users_matrix = LoggedSignals.int_users_matrix;
% Calculate and return reward
H = Ht'*(theta_mat')*Hr + Hd;

SINR = zeros(N_users,1);
for  user_ind = 1 : N_users
    desired = W(:,user_ind)'*H(:,user_ind);
    int_users = int_users_matrix(user_ind,:); % interfering user indices
    interf = [W(:,int_users)'*H(:,user_ind); sqrt(sigma_2)];
    SINR(user_ind) = norm(desired,2)^2/norm(interf,2)^2;
end

%disp(min(SINR))

if min(SINR)>LoggedSignals.SINR_threshold
    Reward = 1/sum(transmit_pow);
    IsDone = 1;
else
    Reward = -1;
    IsDone = 1; %dummy for now .. change later
end

% -------------- Update Logged Signals ------------------
LoggedSignals.chan_index = LoggedSignals.chan_index+1;  % Store new channel index
% LoggedSignals.chan_index
% Fix channel for now: index is always equal to 1
LoggedSignals.new_chan_obs.Ht = LoggedSignals.Ht_mat(:,:,1);
LoggedSignals.new_chan_obs.Hr = LoggedSignals.Hr_mat(:,:,1);
LoggedSignals.new_chan_obs.Hd = LoggedSignals.Hd_mat(:,:,1);
LoggedSignals.State = new_observation;          % Return past state
%LoggedSignals.Action = Action;                 % Return past action
end