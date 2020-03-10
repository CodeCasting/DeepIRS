function [desired_fun, interference_fun, constr_fun] = sinr_CONSTRAINT(V, b, R, SINR_target, sigma_2, all_users, int_users_matrix)
for k = all_users                               % looping over all users
    int_users = int_users_matrix(k,:);          % interfering users
    desired_fun(k) = trace(real(R{k,k}*V)) + square_abs(b{k,k});
    interference_fun(k) = 0;
    for m = int_users
        interference_fun(k) = interference_fun(k) + trace(real(R{k,m}*V)) + square_abs(b{k,m});
    end
    constr_fun(k) = desired_fun(k) - SINR_target * (interference_fun(k) + sigma_2);
end
end