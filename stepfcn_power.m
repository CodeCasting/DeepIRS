function [observation,Reward,IsDone,LoggedSignals] = stepfcn_power(Action, LoggedSignals)
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html
% https://www.mathworks.com/help/reinforcement-learning/ug/define-reward-signals.html

% Input action is taken by the actor network
% This step function calculates the new state/observation and reward
% due to taken (input) action

sigma_2 = LoggedSignals.sigma_2; % noise variance

% Extract current channel from logged signals
Ht = LoggedSignals.new_chan_obs.Ht;
Hr = LoggedSignals.new_chan_obs.Hr;
Hd = LoggedSignals.new_chan_obs.Hd;

N_users = size(Hr,2);
M = size(Ht,1);
N_BS = size(Ht,2);

% Extract BS beamformer from taken action
W = reshape(Action(1:N_BS*N_users)+ 1i*Action(N_BS*N_users+1:2*N_BS*N_users), N_BS, N_users);
% Extract IRS reflection vector from taken action
theta_vec = Action(2*N_BS*N_users+1:2*N_BS*N_users+M)+ 1i*Action(2*N_BS*N_users+M+1:2*(N_BS*N_users+M));
theta_mat = diag(theta_vec);

% Extract past action from Logged signals
past_action = LoggedSignals.Action;

% Calculate transmit power for each user (stacking real and imag powers)
transmit_pow = [diag(real(W)'*real(W)); diag(imag(W)'*imag(W))];

% Calculate received power for each user (also stacking real and imag)
H_W = W'*(Ht'*(theta_mat')*Hr + Hd);
H_real_imag_vec = [real(H_W(:)); imag(H_W(:))];
receive_pow = H_real_imag_vec.^2; 

% Channel observation
chan_obs =  [  real(Ht(:)); imag(Ht(:));
               real(Hr(:)); imag(Hr(:));
               real(Hd(:)); imag(Hd(:))];

observation = [transmit_pow; receive_pow; chan_obs; Action];


% observation = chan_obs;
            
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

if min(SINR)>LoggedSignals.SINR_threshold
    Reward = 1/sum(transmit_pow);
else
    Reward = -1;
end
% dummy for now
IsDone = 1;

new_chan_index = LoggedSignals.chan_index+1;

Hd = 1e-4/sqrt(2)*(randn(N_BS, N_users)+1i*randn(N_BS, N_users));
Hr = 1e-2/sqrt(2)*(randn(M, N_users)+1i*randn(M, N_users));
Ht = 1e-2/sqrt(2)*(randn(M, N_BS)+1i*randn(M, N_BS));

LoggedSignals.new_chan_obs.Ht = Ht;
LoggedSignals.new_chan_obs.Hr = Hr;
LoggedSignals.new_chan_obs.Hd = Hd;

% Update Logged Signals
LoggedSignals.Action = Action;              % Return past action
LoggedSignals.State = observation;          % Return past state
% new_chan_obs.Ht = Ht(new_chan_index);   % check indices
% new_chan_obs.Hr = Hr(new_chan_index);   % check indices
% new_chan_obs.Hd = Hd(new_chan_index);   % check indices
%LoggedSignals.new_chan_obs = new_chan_obs;  % Prepare coming channel
LoggedSignals.chan_index = new_chan_index;  % Store new channel index

end