function [Observation,Reward,IsDone,LoggedSignals] = stepfcn(Action,Ht, Hr, Hd, LoggedSignals)
% https://www.mathworks.com/help/reinforcement-learning/ug/define-reward-signals.html

Observation

Reward

IsDone

LoggedSignals

[real(Ht(:)); imag(Ht(:));
    real(Hr(:)); imag(Hr(:));
    real(Hd(:)); imag(Hd(:))].';

[real(TEMP.W(:)); imag(TEMP.W(:));
real(TEMP.theta(:)); imag(TEMP.theta(:))].';



end