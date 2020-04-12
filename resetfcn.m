% Reset function for the MU-MISO IRS environment
function [InitialObservation,LoggedSignals] = resetfcn()
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
LoggedSignals.chan_index = 1;
InitialObservation = LoggedSignals.State;
end