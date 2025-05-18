clear; clc;

env = PendulumEnv(20, 0.5, pi, false);
validateEnvironment(env)

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

%% Actor and Critic network

baselineNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(1)
    ];

baselineNet = dlnetwork(baselineNet);
baselineNet = initialize(baselineNet);
summary(baselineNet)

baseline = rlValueFunction(baselineNet,obsInfo);

% Define common input path layer
commonPath = [ 
    featureInputLayer(prod(obsInfo.Dimension),Name="comPathIn")
    fullyConnectedLayer(512)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(1,Name="comPathOut") 
    ];

% Define mean value path
meanPath = [
    fullyConnectedLayer(16,Name="meanPathIn")
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension));
    tanhLayer;
    scalingLayer(Name="meanPathOut",Scale=actInfo.UpperLimit) 
    ];

% Define standard deviation path
sdevPath = [
    fullyConnectedLayer(16,"Name","stdPathIn")
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension));
    softplusLayer(Name="stdPathOut") 
    ];

actorNet = dlnetwork(commonPath);
actorNet = addLayers(actorNet,meanPath);
actorNet = addLayers(actorNet,sdevPath);

actorNet = connectLayers(actorNet,"comPathOut","meanPathIn/in");
actorNet = connectLayers(actorNet,"comPathOut","stdPathIn/in");

actorNet = initialize(actorNet);
summary(actorNet)


actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    "ActionMeanOutputNames","meanPathOut",...
    "ActionStandardDeviationOutputNames","stdPathOut",...
    ObservationInputNames="comPathIn");

actor.UseDevice = "gpu";
critic.UseDevice = "gpu";

%% Proximal policy optimization (PPO)

opt = rlTrainingOptions(...
    UseParallel=true,...
    MaxEpisodes=2500,...
    MaxStepsPerEpisode=512,...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480);

gpurng(0)
agent = rlPGAgent(actor,baseline);
agent.SampleTime = env.Ts;
agent.AgentOptions.SampleTime = env.Ts;
agent.AgentOptions.DiscountFactor = 0.99;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 8e-4;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 8e-4;

file_agent = sprintf("PG/agent_damp%.1f_maxtorque%.1f_maxvel%.1f_noise%d.mat", ...
    env.DampingCoefficient, env.MaxTorque, env.AngularVelocityThreshold, env.GaussianNoise);
file_results = sprintf("PG/train_damp%.1f_maxtorque%.1f_maxvel%.1f_noise%d.mat", ...
    env.DampingCoefficient, env.MaxTorque, env.AngularVelocityThreshold, env.GaussianNoise);

% plot(env)
trainResults = train(agent,env,opt);
% save(file_agent,"agent")
% save(file_results,"trainResults")


%% Result

% load(file_agent)
% load(file_results)
simOpts = rlSimulationOptions(MaxSteps=trainResults.TrainingOptions.MaxStepsPerEpisode);
plot(env)
experience = sim(env,agent,simOpts);




