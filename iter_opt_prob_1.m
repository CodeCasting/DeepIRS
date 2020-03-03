function [W, tau, INTERFERENCE, cvx_optval] = iter_opt_prob_1(H,sigma_2,SINR_target,int_users_matrix)

N_BS = size(H,1);
N_users = size(H,2);

%cvx_clear
%clear W tau INTERFERENCE cvx_optval

cvx_begin
cvx_quiet(true)
cvx_solver SDPT3 %SDPT3  %Mosek     % Choose the underlying solver
cvx_precision best        % Change the cvx numerical precision

% Define your optimization variables here
variable W(N_BS,N_users) complex; % add the word binary for binary constraints - see CVX documentation for more options
variable tau nonnegative; % Auxiliary variable
expressions INTERFERENCE(N_users,N_users);

for k = all_users
    int_users = int_users_matrix(k,:); % interfering users
    INTERFERENCE(:,k) = [W(:,int_users)'*H(:,k); sqrt(sigma_2)];
end

% Write the optimization problem
minimize( tau^2 );
subject to
for k = all_users
    {INTERFERENCE(:,k), sqrt(1/SINR_target)*real(((H(:,k)')*W(:,k)))} == complex_lorentz(N_users); % SINR CONSTRAINT
end
{W(:), tau} == complex_lorentz(N_BS * N_users); % POWER CONSTRAINT
cvx_end
end

