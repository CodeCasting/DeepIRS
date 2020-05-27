% Reset function for the MU-MISO IRS environment
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html

function [InitialObservation,LoggedSignals] = resetfcn_power(Hd_mat, Hr_mat, Ht_mat, sigma_2, SINR_threshold)


% Initialize Channel Index to 1
LoggedSignals.chan_index = 1;       % Store new channel index

% Store channels
LoggedSignals.Hd_mat = Hd_mat;
LoggedSignals.Hr_mat = Hr_mat;
LoggedSignals.Ht_mat = Ht_mat;

% Prepare first channels
Hd = LoggedSignals.Hd_mat(:,:,LoggedSignals.chan_index);
Hr = LoggedSignals.Hr_mat(:,:,LoggedSignals.chan_index);
Ht = LoggedSignals.Ht_mat(:,:,LoggedSignals.chan_index);

N_users = size(Hr,2);
M = size(Ht,1);
N_BS = size(Ht,2);

LoggedSignals.new_chan_obs.Ht = Ht;
LoggedSignals.new_chan_obs.Hr = Hr;
LoggedSignals.new_chan_obs.Hd = Hd;

%LoggedSignals.new_chan_obs.Ht = Ht(LoggedSignals.chan_index);   % check indices
%LoggedSignals.new_chan_obs.Hr = Hr(LoggedSignals.chan_index);   % check indices
%LoggedSignals.new_chan_obs.Hd = Hd(LoggedSignals.chan_index);   % check indices


% ---------- Action Initialization ----------
% BS beamforming initialization
W = eye(N_BS, N_users);
W_vec = W(:);
W_realimag_vec = [real(W_vec); imag(W_vec)];
% IRS reflection coefficients initilization
theta_vec = ones(M,1);
theta_realimag_vec = [real(theta_vec); imag(theta_vec)];
theta_mat = diag(theta_vec);
Action = [W_realimag_vec; theta_realimag_vec];


transmit_pow = [diag(real(W)'*real(W)); diag(imag(W)'*imag(W))];

H_W = W'*(Ht'*(theta_mat')*Hr + Hd);
H_real_imag_vec = [real(H_W(:)); imag(H_W(:))];
receive_pow = H_real_imag_vec.^2;

chan_obs =  [  real(Ht(:)); imag(Ht(:));
    real(Hr(:)); imag(Hr(:));
    real(Hd(:)); imag(Hd(:))];

% Return initial environment state variables as logged signals.
LoggedSignals.State =  chan_obs;; %[transmit_pow; receive_pow; chan_obs; Action];
InitialObservation = LoggedSignals.State;

all_users = 1:1:N_users;                    % vector of all user indices

int_users_matrix = meshgrid(all_users).';   % indices of interfering users for each user
int_users_matrix(1:N_users+1:N_users^2) = [];
int_users_matrix = reshape(int_users_matrix, N_users-1, N_users).';
LoggedSignals.int_users_matrix = int_users_matrix;
LoggedSignals.Action = Action;
LoggedSignals.sigma_2 = sigma_2;
LoggedSignals.SINR_threshold = SINR_threshold;

end