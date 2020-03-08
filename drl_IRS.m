function [ACTOR,CRITIC,OPTIONS,agent] = drl_IRS(inputArg1,inputArg2)

%% Function Documentation
% This function implements the algorithm proposed in the paper:

% Chongwen Huang, Ronghong Mo and Chau Yuen, "Reconfigurable Intelligent 
% Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement 
% Learning"


%% 
% Actor Network
    layers = [
        % INPUT Layer
        imageInputLayer([size(XTrain,1),1,1],'Name','input')

        % Fully Connected Layer 1 with Dropout
        fullyConnectedLayer(size(YTrain,3),'Name','fully1')
        tanhLayer('Name','tanh1')
        dropoutLayer(0.5,'Name','dropout1')

        % Fully Connected Layer 2 with Dropout
        fullyConnectedLayer(4*size(YTrain,3),'Name','fully2')
        tanhLayer('Name','tanh2')
        dropoutLayer(0.5,'Name','dropout2')
        
        % OUTPUT Layer
        fullyConnectedLayer(size(YTrain,3),'Name','output')
        regressionLayer('Name','outReg')];
%
%     if Training_Size(dd) < miniBatchSize
%         validationFrequency = Training_Size(dd);
%     else
%         validationFrequency = floor(Training_Size(dd)/miniBatchSize);
%     end
%     VerboseFrequency = validationFrequency;
%
%     % Options
%     options = trainingOptions('sgdm', ...
%         'MiniBatchSize',miniBatchSize, ...
%         'MaxEpochs',20, ...
%         'InitialLearnRate',1e-1, ...
%         'LearnRateSchedule','piecewise', ...
%         'LearnRateDropFactor',0.5, ...
%         'LearnRateDropPeriod',3, ...
%         'L2Regularization',1e-4,...
%         'Shuffle','every-epoch', ...
%         'ValidationData',{XValidation,YValidation}, ...
%         'ValidationFrequency',validationFrequency, ...
%         'Plots','none', ... % 'training-progress'
%         'Verbose',0, ...    % 1
%         'ExecutionEnvironment', 'cpu', ...
%         'VerboseFrequency',VerboseFrequency);


ACTOR

% Critic Network 
CRITIC

% Adjust DDPG options
OPTIONS = rlDDPGAgentOptions('Option1',Value1,'Option2',Value2,...)

% Create DDPG agent
agent = rlDDPGAgent(ACTOR,CRITIC,OPTIONS) 






end

