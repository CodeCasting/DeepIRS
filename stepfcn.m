function [observation,Reward,IsDone,LoggedSignals] = stepfcn(action, LoggedSignals)
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html
% https://www.mathworks.com/help/reinforcement-learning/ug/define-reward-signals.html

% Input action is taken by the actor network
% This step function calculates the new state/observation and reward
% due to taken (input) action

% Extract current channel from logged signals
Ht = LoggedSignals.new_chan_obs.Ht;
Hr = LoggedSignals.new_chan_obs.Hr;
Hd = LoggedSignals.new_chan_obs.Hd;

N_users = size(Hr,2);
M = size(Ht,1);
N_BS = size(Ht,2);

% Extract BS beamformer from taken action
W = reshape(action(1:N_BS*N_users)+ 1i*action(N_BS*N_users+1:2*N_BS*N_users), N_BS, N_users);
% Extract IRS reflection vector from taken action
theta_vec = action(2*N_BS*N_users+1:2*N_BS*N_users+M)+ 1i*action(2*N_BS*N_users+M+1:2*(N_BS*N_users+M));
theta_mat = diag(theta_vec);

% Extract past action from Logged signals
past_action = LoggedSignals.Action;

% Calculate transmit power for each user (stacking real and imag powers)
transmit_pow = [diag(real(W)'*real(W)); diag(imag(W)'*imag(W))];

% Calculate received power for each user (also stacking real and imag)
H = Ht'*(theta_mat')*Hr + Hd;
H_real_imag_vec = [real(H(:)); imag(H(:))];
receive_pow = H_real_imag_vec.^2; 

% Channel observation
chan_obs =  [  real(Ht(:)); imag(Ht(:));
               real(Hr(:)); imag(Hr(:));
               real(Hd(:)); imag(Hd(:))];

observation = [transmit_pow; 
               receive_pow;
               chan_obs;
               past_action];

% Calculate and return reward

Reward = ;

%IsDone = ;

new_chan_index = LoggedSignals.chan_index+1;
new_chan_obs.Ht = Ht(new_chan_index);   % check indices
new_chan_obs.Hr = Hr(new_chan_index);   % check indices
new_chan_obs.Hd = Hd(new_chan_index);   % check indices

% Update Logged Signals
LoggedSignals.Action = action;              % Return past action
LoggedSignals.State = observation;          % Return past state
LoggedSignals.new_chan_obs = new_chan_obs;  % Prepare coming channel
LoggedSignals.chan_index = new_chan_index;  % Store new channel index

end