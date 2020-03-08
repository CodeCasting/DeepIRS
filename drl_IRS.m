function [ACTOR,CRITIC,OPTIONS,agent] = drl_IRS(inputArg1,inputArg2)

%% Function Documentation
% This function implements the algorithm proposed in the paper:

% Chongwen Huang, Ronghong Mo and Chau Yuen, "Reconfigurable Intelligent 
% Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement 
% Learning"


%% 
% Actor Network
ACTOR

% Critic Network 
CRITIC

% Adjust DDPG options
OPTIONS = rlDDPGAgentOptions('Option1',Value1,'Option2',Value2,...)

% Create DDPG agent
agent = rlDDPGAgent(ACTOR,CRITIC,OPTIONS) 






end

