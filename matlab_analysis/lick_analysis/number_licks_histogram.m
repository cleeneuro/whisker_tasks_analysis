% histogram of number of anticipatory licks for CS+ and CS- 
function  number_licks_histogram(csplus_t, csminus_t, reward_t, lick_t, savepath, labels)

%% Constants
SERVO_MOVE_DURATION = 1120;  % ms - duration of servo movement
HISTOGRAM_EDGES = 0:1:15;    % edges for histogram bins
MAX_Y_LIMIT = 400;           % maximum y-axis limit for histograms
CS_PLUS_COLOR = [40/255 53/255 192/255];  % blue color for CS+ histograms
CS_MINUS_COLOR = [0 0 0];    % black color for CS- histograms
CS_PLUS_ALPHA = 0.5;         % transparency for CS+ histograms
CS_MINUS_ALPHA = 0.4;        % transparency for CS- histograms

%% Check the number of input arguments and handle optional saving
    if nargin < 5
        savepath = '';  % Default to no saving if no savepath provided
    end

%% Data processing
% Get dimensions
n_mice = length(csplus_t);
n_days = length(csplus_t{1,1});

% Handle optional labels parameter
if nargin < 6
    % Default labels if not provided
    labels = cell(1, n_days);
    for d = 1:n_days
        labels{d} = ['Day ' num2str(d)];
    end
end

%% CS+ data processing 
for d = 1:n_days
    for m = 1:n_mice
        n_trials = length(csplus_t{m}{d});
        num_licks_d = zeros(1, n_trials);
        for trial = 1:n_trials
            servo_start = csplus_t{m}{d}(trial);
            reward_start = reward_t{m}{d}(trial);   
            num_licks_d(trial) = length(find(lick_t{m}{d}((lick_t{m}{d} > ((servo_start + SERVO_MOVE_DURATION))) & (lick_t{m}{d} < (reward_start))))); 
        end
        
        if m == 1 
            num_licks_d_all = num_licks_d;
        else
            num_licks_d_all = horzcat([num_licks_d_all num_licks_d]);    
        end
    end
    cs_plus_licks{d} = num_licks_d_all; 
end

%% CS- data processing 
% Get an approximation of reward timing relative to texture presentation 
reward_time = mean(reward_t{1}{1} - csplus_t{1}{1});

for d = 1:n_days
    for m = 1:n_mice
        n_trials = length(csminus_t{m}{d});
        num_licks_d = zeros(1, n_trials);
        for trial = 1:n_trials
            servo_start = csminus_t{m}{d}(trial);
            num_licks_d(trial) = length(find(lick_t{m}{d}((lick_t{m}{d} > ((servo_start + SERVO_MOVE_DURATION))) & (lick_t{m}{d} < (servo_start + reward_time))))); 
        end
        
        if m == 1 
            num_licks_d_all = num_licks_d;
        else
            num_licks_d_all = horzcat([num_licks_d_all num_licks_d]);    
        end
    end
    cs_minus_licks{d} = num_licks_d_all; 
end
%% Create visualization 
close all
h = length(findobj('type','figure'));

% Create figure with tiled layout
fh = figure(h+1); 
tiledlayout(2, ceil(n_days/2), 'TileSpacing', 'compact', 'Padding', 'compact');

for d = 1:n_days 
    nexttile(d)
    h1 = histogram(cs_plus_licks{d}, HISTOGRAM_EDGES);
    h1.FaceColor = CS_PLUS_COLOR;
    h1.FaceAlpha = CS_PLUS_ALPHA;
    h1.EdgeColor = [0 0 0];
    hold on 
    h2 = histogram(cs_minus_licks{d}, HISTOGRAM_EDGES);
    h2.FaceColor = CS_MINUS_COLOR;
    h2.FaceAlpha = CS_MINUS_ALPHA; 
    h2.EdgeColor = [0 0 0];
    ylim([0 MAX_Y_LIMIT])
    
    % Set title using provided labels
    title(labels{d})
    
    xlabel('Number of anticipatory licks')
    ylabel('Count')
end

% Add legend
text(12, 350, 'CS+','Color', CS_PLUS_COLOR)
text(12, 300, 'CS-')

% Set overall title and figure properties
sgtitle('Number of Licks')
fh.Position = [52 279 1420 445];    % Change to a position where it's not squished

% Save figure if savepath is provided and not empty
if ~isempty(savepath)
    filename = 'NumberLicksHistogram';
    full_savepath = fullfile(savepath, [filename, '.jpeg']);
    saveas(figure(h+1), full_savepath);
    fprintf('Figure saved to: %s\n', full_savepath);
end