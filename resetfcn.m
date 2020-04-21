% Reset function for the MU-MISO IRS environment
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html

function [InitialObservation,LoggedSignals] = resetfcn()

% Initialize Channel Index to 1
LoggedSignals.chan_index = 1;       % Store new channel index
% Prepare first channels
LoggedSignals.new_chan_obs.Ht = Ht(LoggedSignals.chan_index);   % check indices
LoggedSignals.new_chan_obs.Hr = Hr(LoggedSignals.chan_index);   % check indices
LoggedSignals.new_chan_obs.Hd = Hd(LoggedSignals.chan_index);   % check indices

transmit_pow
receive_pow
chan_obs

% ---------- Action Initialization ----------
% BS beamforming initialization
W_vec = ones(2*N_BS*N_users,1);
% IRS reflection coefficients initilization
theta_vec = ones(2*M,1);
action = [W_vec; theta_vec];

% Return initial environment state variables as logged signals.
LoggedSignals.State = [transmit_pow; receive_pow; chan_obs; action];
InitialObservation = LoggedSignals.State;
end