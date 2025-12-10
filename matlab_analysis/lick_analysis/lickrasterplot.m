 %% Make Licking Raster Plots 
% This function creates raster plots for licking behavior during CS+ and CS- trials
% 
% INPUTS:
%   csplus_t    - Cell array of CS+ trial times for each mouse and day
%   csminus_t   - Cell array of CS- trial times for each mouse and day  
%   reward_t    - Cell array of reward times for each mouse and day
%   lick_t      - Cell array of lick times for each mouse and day
%   mousenames  - Cell array of mouse names
%   labels      - Cell array of day labels (inherited from Textfile_lickanalysis)
%   varargin    - Optional save path for figure output
%
% The labels parameter inherits day/session labels from Textfile_lickanalysis.m
% If labels are missing or incorrect length, default "Day X" labels are generated.
%
function  lickrasterplot(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels, varargin)

%% Check the number of input arguments
    if nargin > 6
        % Handle optional input arguments
        savepath = varargin{1};
    end

%% Validate and prepare labels
    n_mice = length(csplus_t);
    if n_mice > 0
        n_days = length(csplus_t{1});
        
        % Check if labels are provided and have correct length
        if isempty(labels) || length(labels) ~= n_days
            warning('Labels not provided or incorrect length. Generating default labels.');
            labels = cell(1, n_days);
            for d = 1:n_days
                labels{d} = sprintf('Day %d', d);
            end
        end
    else
        error('No data provided for analysis.');
    end

% set time window for plot
 tStart = -500; % ms before the trial start 
 tEnd = 3000;   % ms after reward
%% CS+ plot 
 for m = 1:n_mice
    n_days = length(csplus_t{1,m});
    
    % Calculate number of figures needed (max 5 subplots per figure)
    max_subplots_per_figure = 5;
    n_figures = ceil(n_days / max_subplots_per_figure);
    
    % Initialize figure counter
    fig_counter = 0;
    current_fig = 0;
     
     for d = 1:n_days
         n_trials = length(csplus_t{m}{d});
         for trial = 1:n_trials
             servo_start = csplus_t{m}{d}(trial);
             reward_start = reward_t{m}{d}(trial);
             %reward_start = csplus_t{m}{d}(trial) + 2900; % hard coded because I accidentally changed textfile format for reward print statements
             trial_lick_t = [];
             trial_lick_t = lick_t{m}{d}((lick_t{m}{d} > ((servo_start+tStart))) & (lick_t{m}{d} < (reward_start+tEnd)));
             trial_lick_t = trial_lick_t - servo_start;   % subtract texture_start to make texture_start time = 0s
             if trial == 1 %concatenate the lick times across trials 
                 trial_lick_t_cat = trial_lick_t;
                 %subtract trial from n_trials so that trial 1 is plotted
                 %at the top and trial n is at the bottom. Add 1 so the
                 %first trial is 1 and not 0
                 y_cat = ones(1,length(trial_lick_t))*(n_trials+1-trial);   
             else
                 trial_lick_t_cat = vertcat(trial_lick_t_cat, trial_lick_t);
                 y_cat = horzcat([y_cat, ones(1,length(trial_lick_t))*(n_trials+1-trial)]);
             end
         end
         
         rewardTime = reward_start - servo_start;
         % rewardTime = (reward_t{m}{d}(trial) - csplus_t{m}{d}(trial));

         % Determine which figure this day belongs to
         fig_index = ceil(d / max_subplots_per_figure);
         subplot_index = mod(d - 1, max_subplots_per_figure) + 1;
         
         % Create new figure if needed
         if fig_index ~= current_fig
             current_fig = fig_index;
             fh = figure(m * 100 + fig_index);  % Unique figure number
             fh.Position = [18 50 800 1600];  % Taller figure size to reduce horizontal squishing
             % Add title to each figure
             sgtitle([mousenames{m}, ': CS+'])
         end
         
         % Create subplot (max 5 per figure)
         subplot(max_subplots_per_figure, 1, subplot_index)
            % Add more padding and space from title
            set(gca, 'Position', get(gca, 'Position') + [0.05 0.02 -0.1 -0.05]);
             sz = 4;
             s = scatter(trial_lick_t_cat,y_cat,sz,'k','filled');
             s.MarkerFaceAlpha = 1;
             % Use inherited label from Textfile_lickanalysis
             if d <= length(labels)
                 title(labels{d});
             else
                 title(sprintf('Day %d', d));
             end

             stim_in = 1120;    
             stim_out = stim_in + 2000;
             axis([tStart rewardTime+tEnd 0 n_trials+1])
             xline(rewardTime)
             % text((reward_start - texture_start), n_trials+5, 'Reward')
             v = [stim_in 0; stim_in n_trials+1; stim_out n_trials+1; stim_out 0];
             f = [1 2 3 4];
             p_l = patch('Faces',f,'Vertices',v,'Facecolor', [0.5 .3 0.6]);
             p_l.FaceAlpha = 0.3;
             p_l.EdgeColor = 'none';
             % text(stim_in, n_trials+5, 'CS+', 'FontSize',9)
             triallabels = n_trials-100:20:n_trials;
             % triallabels = [0 n_trials];   
             yticks(triallabels)
             yticklabels({'100', '80', '60', '40','20', '0'});
             % yticklabels({num2str(n_trials) '1'})
             xlabel('Time from trial start', 'FontSize',10)
             ylabel('Trial Number', 'FontSize',10)         
     end

   if nargin > 6
            % Save all figures for this mouse
            for fig_idx = 1:n_figures
                filename = strcat('CS_plus_raster_',mousenames{m},'_fig',num2str(fig_idx));
                % Save as high-resolution PNG
                print(figure(m * 100 + fig_idx), strcat(savepath,filesep,filename,'.png'), '-dpng', '-r300');
            end
    end
 end


%% CS- plot 
 for m = 1:n_mice
    n_days = length(csplus_t{1,m});
    
    % Calculate number of figures needed (max 5 subplots per figure)
    max_subplots_per_figure = 5;
    n_figures = ceil(n_days / max_subplots_per_figure);
    
    % Initialize figure counter
    fig_counter = 0;
    current_fig = 0;
     
     for d = 1:n_days
         n_trials = length(csminus_t{m}{d});
         for trial = 1:n_trials
             servo_start = csminus_t{m}{d}(trial);
             trial_lick_t = [];
             trial_lick_t = lick_t{m}{d}((lick_t{m}{d} > ((servo_start+tStart))) & (lick_t{m}{d} < (servo_start+3250 + tEnd))); %reward start is typically 3250ms after texture start
             trial_lick_t = trial_lick_t - servo_start;
             if trial == 1
                 trial_lick_t_cat = trial_lick_t;
                 y_cat = ones(1,length(trial_lick_t))*(n_trials+1-trial);
             else
                 trial_lick_t_cat = vertcat(trial_lick_t_cat, trial_lick_t);
                 y_cat = horzcat([y_cat, ones(1,length(trial_lick_t))*(n_trials+1-trial)]);
             end
         end
            % Determine which figure this day belongs to
            fig_index = ceil(d / max_subplots_per_figure);
            subplot_index = mod(d - 1, max_subplots_per_figure) + 1;
            
            % Create new figure if needed
            if fig_index ~= current_fig
                current_fig = fig_index;
                fh = figure(n_mice * 100 + m * 100 + fig_index);  % Unique figure number for CS-
                fh.Position = [18 50 800 1600];  % Taller figure size to reduce horizontal squishing
                % Add title to each figure
                sgtitle([mousenames{m}, ': CS-'])
            end
            
            % Create subplot (max 5 per figure)
            subplot(max_subplots_per_figure, 1, subplot_index)
            % Add more padding and space from title
            set(gca, 'Position', get(gca, 'Position') + [0.05 0.02 -0.1 -0.05]);
             sz = 4;
             s = scatter(trial_lick_t_cat,y_cat,sz,'k','filled');
             % Use inherited label from Textfile_lickanalysis
             if d <= length(labels)
                 title(labels{d});
             else
                 title(sprintf('Day %d', d));
             end
             stim_in = 1120; % this is found empirically b/c we dont know actual stim in time. The "arduino" time it takes based on our measures of stim in to reward time using camera
             stim_out = stim_in + 2000; % we know based on camera that stim is in for 4s
             axis([tStart rewardTime+tEnd 0 n_trials+1])
             xline(rewardTime,'--k')
             v = [stim_in 0; stim_in n_trials+1; stim_out n_trials+1; stim_out 0];
             f = [1 2 3 4];
             p_l = patch('Faces',f,'Vertices',v,'Facecolor', [0.5 .3 0.6]);
             p_l.FaceAlpha = 0.3;
             p_l.EdgeColor = 'none';

             % patch for 

             % text(stim_in, n_trials+5, 'CS-', 'FontSize',9)
             % triallabels = [0 n_trials];   
             % yticks(triallabels)
             % yticklabels({num2str(n_trials) '1'})
             yticks(triallabels)
             yticklabels({'100', '80', '60', '40','20', '0'});
             xlabel('Time from trial start', 'FontSize',10)
             ylabel('Trial Number', 'FontSize',10)

     end

    if nargin > 6
       % Save all figures for this mouse
       for fig_idx = 1:n_figures
           filename = strcat('CS_minus_raster_',mousenames{m},'_fig',num2str(fig_idx));
           % Save as high-resolution PNG
           print(figure(n_mice * 100 + m * 100 + fig_idx), strcat(savepath,filesep,filename,'.png'), '-dpng', '-r300');
       end
    end
 end



