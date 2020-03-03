function [V, a_aux, a, b, R, desired, interference, SINR_CONSTR, cvx_optval] = iter_opt_prob_2(W, Ht,Hr,Hd,sigma_2,SINR_target,int_users_matrix)

N_users = size(Hd,2);
M = size(Ht,1);

all_users = 1:N_users;

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
end

