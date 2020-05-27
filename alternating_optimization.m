%disp('Running alternating optimization algorithm')
% r=1;            % iteration index
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
%     r = r+1;
end