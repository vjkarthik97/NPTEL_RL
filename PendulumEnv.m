classdef PendulumEnv < rl.env.MATLABEnvironment
    %PENDULUM: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        % Acceleration due to gravity in m/s^2
        Gravity = 9.81
        
        % Mass of the pendulum
        PendulumMass = 1
        
        % Length of the pendulum
        PendulumLength = 1

        % Coefficient of air friction damping
        DampingCoefficient = 0.5
        
        % Max Torque and Torque the input can apply
        MaxTorque = 1
        Torque = 0
               
        % Sample time
        Ts = 0.05
        T = 0
        
        % Anglulr velocity at which to fail the episode (radians/s)
        AngularVelocityThreshold = 8
        
        % Noise
        GaussianNoise = false

        % Reward each time step the cart-pole is balanced
        RewardForNotFalling = 1
        
        % Penalty when the cart-pole fails to balance
        PenaltyForFalling = -10 
    end
    
    properties
        % Initialize system state [cos(theta),sin(theta),dtheta]'
        State = zeros(3,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = PendulumEnv(max_torque, damping_coeff, max_angular_vel, gauss_noise)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([3 1]);
            ObservationInfo.Name = 'Pendulum States';
            ObservationInfo.Description = 'costheta, sintheta, dtheta';

            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec(1);
            ActionInfo.Name = 'Pendulum Action';
            ActionInfo.Description = 'torque';
            
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
            updateObservationInfo(this);

            this.MaxTorque = max_torque;
            this.DampingCoefficient = damping_coeff;
            this.AngularVelocityThreshold = max_angular_vel;
            this.GaussianNoise = gauss_noise;
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];

            % Noise
            if this.GaussianNoise == true
                noise = normrnd(0,sqrt(0.001), 2,1);
            else
                noise = [0;0];
            end
            
            % disp(this.State)
            % Unpack state vector
            CosTheta = this.State(1); SinTheta = this.State(2);
            Theta = atan2(SinTheta, CosTheta) + noise(1);
            ThetaDot = this.State(3)*this.AngularVelocityThreshold + noise(2);
            

            % this.Torque = 2*(pi*sign(ThetaDot) - Theta) - 1*ThetaDot;% + this.PendulumMass*this.Gravity*this.PendulumLength*SinTheta;
            % Get action
            this.Torque = getForce(this,Action);
            this.Torque = min(max(this.Torque, -1), 1);

            % Apply motion equations            
            ThetaDotDot = - this.Gravity/this.PendulumLength* SinTheta - this.DampingCoefficient/(this.PendulumMass*this.PendulumLength^2)*ThetaDot ...
                          + 1/(this.PendulumMass*this.PendulumLength^2)*this.Torque*this.MaxTorque;
            
            % Euler integration
            new_ThetaDot = ThetaDot +  this.Ts * ThetaDotDot;
            new_ThetaDot = min(max(new_ThetaDot/this.AngularVelocityThreshold, -1), 1);
            new_Theta = Theta + this.Ts * new_ThetaDot*this.AngularVelocityThreshold;
            Observation = [cos(new_Theta); sin(new_Theta); new_ThetaDot];

            % Update system states
            this.State = Observation;
            this.T = this.T + this.Ts; 

            
            % Check terminal condition
            IsDone = false;
            this.IsDone = IsDone;

            % Get reward
            Reward = getReward(this);

            % fprintf("time: %.2f, error: %.2f, rate: %.2f, torque: %.2f, reward: %.4f\n", ...
            %     this.T, rad2deg(abs(pi - abs(new_Theta))), new_ThetaDot*this.AngularVelocityThreshold, this.Torque, Reward)

            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            % Theta (+- .05 rad)
            T0 = (2 * rand - 1)*0.05;  
            % Thetadot
            Td0 = 0;
            
            InitialObservation = [cos(T0); sin(T0);Td0];
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        function torque = getForce(this,action)
            torque = min(max(action, -this.MaxTorque), this.MaxTorque);           
        end
        % update the action info based on max force
        function updateActionInfo(this)
            this.ActionInfo.LowerLimit = -1; 
            this.ActionInfo.UpperLimit = 1;
        end
        
        function updateObservationInfo(this)
            this.ObservationInfo.LowerLimit = [-1; -1; -1]; 
            this.ObservationInfo.UpperLimit = [1; 1; 1];
        end

        
        
        % Reward function
        function Reward = getReward(this)
            % if ~this.IsDone
            %     Reward = this.RewardForNotFalling;
            % else
            %     Reward = this.PenaltyForFalling;
            % end     
            Theta = atan2(this.State(2), this.State(1));
            ThetaDot = this.State(3);

            Reward = -(0.5*(abs(Theta)/pi - 1)^2 ...
                      + 0.25*(ThetaDot)^2 ...
                      + 0.25*(this.Torque)^2);
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            this.Figure = figure('Visible','on','HandleVisibility','off');
            
            ha = gca(this.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.XLim = [-this.PendulumLength-0.2 this.PendulumLength+0.2];
            ha.YLim = [-this.PendulumLength-0.2 this.PendulumLength+0.2];
            hold(ha,'on');
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',3},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        function set.PendulumLength(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','HalfPoleLength');
            this.PendulumLength = val;
            notifyEnvUpdated(this);
        end
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Gravity');
            this.Gravity = val;
        end
        function set.PendulumMass(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','CartMass');
            this.PendulumMass = val;
        end
        function set.MaxTorque(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','MaxForce');
            this.MaxTorque = val;
            updateActionInfo(this);
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        function set.AngularVelocityThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','AngularVelocityThreshold');
            this.AngularVelocityThreshold = val;
        end
        function set.DampingCoefficient(this,val)
            validateattributes(val,{'numeric'},{'finite','real','nonnegative','scalar'},'','DampingCoefficient');
            this.DampingCoefficient = val;
        end
        function set.RewardForNotFalling(this,val)
            validateattributes(val,{'numeric'},{'real','finite','scalar'},'','RewardForNotFalling');
            this.RewardForNotFalling = val;
        end
        function set.PenaltyForFalling(this,val)
            validateattributes(val,{'numeric'},{'real','finite','scalar'},'','PenaltyForFalling');
            this.PenaltyForFalling = val;
        end
    end
    properties (Access = protected)
        % Handle to figure
        Figure
    end

    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                % Set visualization figure as the current figure
                ha = gca(this.Figure);

                % Extract the cart position and pole angle
                CosTheta = this.State(1); SinTheta = this.State(2);
                Theta = atan2(SinTheta, CosTheta) - pi/2;
                L = this.PendulumLength;
                
                pts = [0, 0; 
                       L*cos(Theta), L*sin(Theta)];

                poleplot = findobj(ha,'Tag','poleplot');
                bobplot = findobj(ha,'Tag','bobplot');
                if isempty(poleplot) || ~isvalid(poleplot)
                    % Initialize the pendulum plot
                    poleplot = plot(ha,pts(:,1), pts(:,2), '-b', 'LineWidth',3);
                    poleplot.Tag = 'poleplot';
                    
                    % Initialize the pendulum plot
                    bobplot = scatter(ha,pts(2,1), pts(2,2), 500, 'r', 'filled');
                    bobplot.Tag = 'bobplot';
                else
                    set(poleplot,'XData',pts(:,1),'YData',pts(:,2))
                    set(bobplot,'XData',pts(2,1),'YData',pts(2,2))
                end

                

                % Refresh rendering in the figure window
                drawnow();
            end
        end
    end
end
