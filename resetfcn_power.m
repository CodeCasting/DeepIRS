% Reset function for the MU-MISO IRS environment
% https://www.mathworks.com/help/reinforcement-learning/ug/create-custom-reinforcement-learning-environment-in-matlab.html

function [InitialObservation,LoggedSignals] = resetfcn_power(Hd_mat, Hr_mat, Ht_mat, sigma_2, SINR_threshold, state_design)

% -------------- Initialized Logged Signals ------------------
LoggedSignals.chan_index = 1;                                   % Initialize channel/step index
LoggedSignals.StateDes = state_design;
LoggedSignals.sigma_2 = sigma_2;                                % Noise variance for all users
LoggedSignals.SINR_threshold = SINR_threshold;                  % SINR threshold for all users
% Store ALL training channels in Logged Signals (3-dimensional matrices)
LoggedSignals.Hd_mat = Hd_mat;
LoggedSignals.Hr_mat = Hr_mat;
LoggedSignals.Ht_mat = Ht_mat;

% Prepare first channels
Hd = LoggedSignals.Hd_mat(:,:,LoggedSignals.chan_index);
Hr = LoggedSignals.Hr_mat(:,:,LoggedSignals.chan_index);
Ht = LoggedSignals.Ht_mat(:,:,LoggedSignals.chan_index);

N_users = size(Hr,2); M = size(Ht,1); N_BS = size(Ht,2);

LoggedSignals.new_chan_obs.Ht = Ht;
LoggedSignals.new_chan_obs.Hr = Hr;
LoggedSignals.new_chan_obs.Hd = Hd;


% ---------- Action Initialization ----------
% BS beamforming initialization
W = eye(N_BS, N_users);
W_vec = W(:);
W_realimag_vec = [real(W_vec); imag(W_vec)];
% IRS reflection coefficients initilization
%theta_vec = ones(M,1); %theta_realimag_vec = [real(theta_vec); imag(theta_vec)]; %theta_mat = diag(theta_vec);
theta_vec = pi*ones(M,1);
theta_mat = diag(exp(1i*theta_vec));
action = [W_realimag_vec; theta_vec];
% Real/imaginary transmitted power for each user
transmit_pow = [diag(real(W)'*real(W)); diag(imag(W)'*imag(W))];

% Real/imaginary received power at each user
H_W = W'*(Ht'*(theta_mat')*Hr + Hd);
H_real_imag_vec = [real(H_W(:)); imag(H_W(:))];
receive_pow = H_real_imag_vec.^2;
% Channel observation vector
chan_obs = [real(Ht(:)); imag(Ht(:)); real(Hr(:)); imag(Hr(:)); real(Hd(:)); imag(Hd(:))];


% Calculate interferer indices matrix
int_users_matrix = meshgrid(1:N_users).';
int_users_matrix(1:N_users+1:N_users^2) = [];
int_users_matrix = reshape(int_users_matrix, N_users-1, N_users).';


% -------------- Initialized Logged Signals ------------------
switch LoggedSignals.StateDes
    case 1
        LoggedSignals.State =  [transmit_pow; receive_pow; chan_obs; action];
    case 2
        LoggedSignals.State =  chan_obs;
end
LoggedSignals.int_users_matrix = int_users_matrix;              % Interferer indices matrix
% LoggedSignals.action = action;

% Return initial observation
InitialObservation = LoggedSignals.State;

end