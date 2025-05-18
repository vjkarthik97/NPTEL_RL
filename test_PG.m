clear; clc;

env = PendulumEnv(20, 0.5, 4, false);
validateEnvironment(env)

file_agent = sprintf("PG/agent_damp%.1f_maxtorque%.1f_maxvel%.1f_noise%d.mat", ...
    env.DampingCoefficient, env.MaxTorque, env.AngularVelocityThreshold, env.GaussianNoise);
file_results = sprintf("PG/train_damp%.1f_maxtorque%.1f_maxvel%.1f_noise%d.mat", ...
    env.DampingCoefficient, env.MaxTorque, env.AngularVelocityThreshold, env.GaussianNoise);

env.GaussianNoise = true;
load(file_agent)
load(file_results)
simOpts = rlSimulationOptions(MaxSteps=trainResults.TrainingOptions.MaxStepsPerEpisode);
plot(env)
experience = sim(env,agent,simOpts);

actor = getActor(agent);
critic = getCritic(agent);
actorNet = getModel(actor);
criticNet = getModel(critic);
figure("Name", "actor")
plot(actorNet)
figure("Name", "critic")
plot(criticNet)

fig1 = figure("Name","train");
plot(trainResults.EpisodeIndex, trainResults.AverageReward)
ylabel("Average Reward")
xlabel("Episodes")
title("PG training")
grid on
exportgraphics(fig1, "PG/train.pdf")


fig2 = figure("Name", "control");
theta = atan2(experience.Observation.PendulumStates.Data(2,:,:), ...
              experience.Observation.PendulumStates.Data(1,:,:));
theta = squeeze(rad2deg(theta));
for i=20:length(theta)
    if theta(i-1) < 0 && theta(i) > 0
        % disp([i,theta(i)])
        theta(i) = theta(i) -360 ;

    elseif theta(i-1) > 0 && theta(i) < 0
        % disp([i,theta(i)])
        theta(i) = theta(i) + 360 ;
    end
end
omega = squeeze(rad2deg(experience.Observation.PendulumStates.Data(3,:,:)));
control = squeeze(experience.Action.PendulumAction.Data)*env.MaxTorque;

sgtitle("vertical stabilization from \theta_0 = "+theta(1) + "(deg)")
subplot(3,1,1)
plot(experience.Observation.PendulumStates.Time, theta,  'LineWidth',2)
hold on
plot([experience.Observation.PendulumStates.Time(1), experience.Observation.PendulumStates.Time(end)], ...
    [180*sign(theta(end)), 180*sign(theta(end))], '--k', 'LineWidth',2)
ylabel("\theta (deg)")
xlim([0, experience.Observation.PendulumStates.Time(end)])
grid on

subplot(3,1,2)
plot(experience.Observation.PendulumStates.Time, omega, 'LineWidth',2)
ylabel("\omega (deg/s)")
xlim([0, experience.Observation.PendulumStates.Time(end)])
grid on

subplot(3,1,3)
plot(experience.Action.PendulumAction.Time, control, 'LineWidth',2)
xlim([0, experience.Action.PendulumAction.Time(end)])
ylabel("Torque (\tau) (Nm)")
xlabel("Time (s)")
grid on


filefig = sprintf("PG/initTheta%.1f.pdf", theta(1));
exportgraphics(fig2, filefig)



