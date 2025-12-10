% Create a figure for each mouse that consists of subplots, each with that mouse's daily
% average lick rate (averaged across trials) for CS minus trials in black and CS plus trials in blue

function lick_rate_plot_mouse_all_days(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels, varargin)
%% Check the number of input arguments
    if nargin > 6
        % Handle optional input arguments
        savepath = varargin{1};
    end
close all

% Constants
PRE_TRIAL_TIME_MS = 0;  % Time before trial start (changed from -1500)
POST_REWARD_TIME_MS = 6000;  % Time after reward
BIN_SIZE_MS = 200;  % Bin size for histogram
STIMULUS_DURATION_MS = 2000;  % Duration of stimulus presentation
SERVO_MOVE_TIME_MS = 1120;  % Time for servo movement (empirically determined)
TIME_TO_REWARD_MS = 2000;  % Time from linear actuator start to reward
Y_AXIS_MAX = 15;  % Maximum value for y-axis

% Derived constants
multiplier = 1000 / BIN_SIZE_MS;  % Convert to licks per second
stim_in = SERVO_MOVE_TIME_MS;  % Stimulus start time
stim_out = stim_in + STIMULUS_DURATION_MS;  % Stimulus end time

% Calculate time window and edges
edges = PRE_TRIAL_TIME_MS:BIN_SIZE_MS:TIME_TO_REWARD_MS + POST_REWARD_TIME_MS;

% Calculate stimulus bins for analysis
binStart = ceil((stim_in - PRE_TRIAL_TIME_MS) / BIN_SIZE_MS);
binEnd = floor((stim_out - PRE_TRIAL_TIME_MS) / BIN_SIZE_MS);
stimulusBins = binStart:binEnd;

n_mice = length(csplus_t);

 
    
% Process data for each mouse
for m = 1:n_mice
    n_days(m) = size(csplus_t{m},2);
    for d = 1:n_days(m)
         plus_trials = length(csplus_t{m}{d});
         minus_trials = length(csminus_t{m}{d}); 


         % Initialize lick histograms based on the length of edges
         lickhist_plus = zeros(plus_trials, length(edges) - 1);
         lickhist_minus = zeros(minus_trials, length(edges) - 1);

         % Process CS+ trials
         for trial = 1:plus_trials
             servo_start = csplus_t{m}{d}(trial);
             reward_start = reward_t{m}{d}(trial);
             trial_lick_plus = lick_t{m}{d}((lick_t{m}{d} > (servo_start + PRE_TRIAL_TIME_MS)) & (lick_t{m}{d} < (reward_start + POST_REWARD_TIME_MS)));          
             trial_lick_plus = trial_lick_plus - servo_start;
             lickhist_plus(trial,:) = histcounts(trial_lick_plus, edges);
         end
         
         lickhist_plus = lickhist_plus .* multiplier; % Scale to licks/s 
       
         % Process CS- trials
         for trial = 1:minus_trials
             servo_start = csminus_t{m}{d}(trial); 
             trial_lick_minus = lick_t{m}{d}((lick_t{m}{d} > (servo_start + PRE_TRIAL_TIME_MS)) & (lick_t{m}{d} < (servo_start + STIMULUS_DURATION_MS + POST_REWARD_TIME_MS)));
             trial_lick_minus = trial_lick_minus - servo_start;
             lickhist_minus(trial,:) = histcounts(trial_lick_minus, edges);
         end
        
         lickhist_minus = lickhist_minus .* multiplier;
    
         % Calculate averages for each mouse
         avg_lickhist_minus{m}(d,:) = mean(lickhist_minus, 1);
         avg_lickhist_plus{m}(d,:) = mean(lickhist_plus, 1);
         
         % Calculate standard error of the mean from across trials 
         std_plus{m}(d,:) = std(lickhist_plus, 1) ./ sqrt(plus_trials);
         std_minus{m}(d,:) = std(lickhist_minus, 1) ./ sqrt(minus_trials);

         %% Test for significance 
        %  % Extract data from the specified time bins across all trials
        %  stim_plus_bins = lickhist_plus(:, stimulusBins);
        %  stim_minus_bins = lickhist_minus(:, stimulusBins);
        % 
        %  % Flatten the matrices to create vectors for each condition
        %  plus_flattened = stim_plus_bins(:);
        %  minus_flattened = stim_minus_bins(:);
        % 
        %  fprintf('day %s\n', num2str(d));
        % 
        % [~, p1] = lillietest(plus_flattened);
        % fprintf('Normality test for CS plus trials: p = %.3f\n', p1);
        % 
        % [~, p2] = lillietest(minus_flattened);
        % fprintf('Normality test for CS minus trials: p = %.3f\n', p2);
        % 
        % % Perform the appropriate test
        % if p1 > 0.05 && p2 > 0.05
        %     % Data is normally distributed, use paired t-test
        %     [~, pValue, ci, stats] = ttest(plus_flattened, minus_flattened);
        %     fprintf('Paired t-test: t(%d) = %.2f, p = %.3f\n', stats.df, stats.tstat, pValue);
        % else
        %     % Data is not normally distributed, use Wilcoxon signed-rank test
        %     pValue = signrank(plus_flattened, minus_flattened);
        %     fprintf('Wilcoxon signed-rank test: p = %.3f\n', pValue);
        % end
    end    

    rewardTime = mean(reward_t{m}{d} - csplus_t{m}{d});
    
    centres = edges(1:end-1) + diff(edges)/2;
    centres_s = centres ./ 1000;
    h = length(findobj('type','figure'));
end

% Create plots for each mouse
create_mouse_plots(avg_lickhist_plus, avg_lickhist_minus, std_plus, std_minus, ...
    labels, mousenames, n_mice, n_days, centres, rewardTime, stim_in, stim_out, ...
    PRE_TRIAL_TIME_MS, POST_REWARD_TIME_MS, Y_AXIS_MAX, h, nargin, savepath);

function create_mouse_plots(avg_lickhist_plus, avg_lickhist_minus, std_plus, std_minus, ...
    labels, mousenames, n_mice, n_days, centres, rewardTime, stim_in, stim_out, ...
    tStart, tEnd, yMax, h, nargin, savepath)
% Create and format plots for each mouse

for m = 1:n_mice
    % Make one figure per mouse
    fh = figure(m + h);

    % Set up tiled layout: 1 row, n_days(m) columns
    t = tiledlayout(1, n_days(m), ...
        'TileSpacing', 'compact', ... % Removes gaps between tiles
        'Padding', 'none');           % Removes outer whitespace

    for d = 1:n_days(m)
        ax = nexttile;

        % Plot lick rate data
        shadedErrorBar_cl(centres, avg_lickhist_plus{m}(d,:), std_plus{m}(d,:));
        hold on
        shadedErrorBar(centres, avg_lickhist_minus{m}(d,:), std_minus{m}(d,:));
        title(labels{d});

        % Set up x-axis ticks
        xtickValues = tStart:1000:tEnd;
        xticks(xtickValues);
        xtickLabels = (xtickValues - tStart) / 1000;
        xticklabels(xtickLabels);
        ax.XTickLabelRotation = 0;

        % Set axis limits
        axis([tStart tEnd 0 yMax]);

        % Add reference lines and stimulus patch
        xline(rewardTime, '--k');
        v = [stim_in 0; stim_in yMax; stim_out yMax; stim_out 0];
        p_l = patch('Faces', [1 2 3 4], 'Vertices', v, 'Facecolor', [0.5 0.5 0.5]);
        p_l.FaceAlpha = 0.3;
        p_l.EdgeColor = 'none';
        text(stim_in, yMax - 0.5, 'Texture', 'FontSize', 9)
    end

    % Add shared labels
    xlabel(t, 'Time from trial start (s)');
    ylabel(t, 'Lick rate (licks/s)');

    % Add legend text to the last tile
    nexttile(n_days(m));
    text(4200, yMax - 2, 'CS+', 'Color', [40 53 192]/255);
    text(4200, yMax - 3, 'CS-');

    % Set overall title
    sgtitle(mousenames{m});

    % Adjust figure size and position
    fh.Position = [65 475 1445 270];
    
    % Save figure if savepath is provided
    if nargin > 6
        filename = strcat('Avg_Lickrate_', mousenames{m});
        full_filepath = strcat(savepath, filesep, filename, '.jpeg');
        saveas(figure(m + h), full_filepath)
        disp('Figure saved at:')
        disp(full_filepath)
    end
end

