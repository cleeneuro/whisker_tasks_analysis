function lick_rate_plot(csplus_t, csminus_t, reward_t, lick_t, labels, varargin)

%% Check the number of input arguments
    if nargin > 5
        % Handle optional input arguments
        savepath = varargin{1};
    end
close all

% Constants
stimulus_duration = 2000; % ms - duration of stimulus presentation
stim_in_time = 1120; % ms - empirically found stimulus start time

% set time window for plot
 t_start = -1500; % ms before the trial start 
 t_end = 5000;   % ms after reward 
 time_to_reward = 3250; % time from linear act start to reward time (as measured by the arduino)
 n_mice = length(csplus_t);
 n_days = length(csplus_t{1,1});  
 bin_size = 200; % in ms 
 multiplier = 1000./bin_size;
 edges = t_start:bin_size:time_to_reward+t_end;

% Process data and create plots
[centres, reward_time, daily_avg_lickhist_plus, daily_avg_lickhist_minus, std_plus, std_minus] = process_lick_data(csplus_t, csminus_t, reward_t, lick_t, edges, t_start, t_end, stimulus_duration, multiplier, n_mice, n_days);
create_lick_rate_plots(centres, daily_avg_lickhist_plus, daily_avg_lickhist_minus, std_plus, std_minus, labels, reward_time, stim_in_time, stimulus_duration, t_start, t_end, n_days);

if nargin > 5
    filename = strcat('avg_across_mice_lickrate');
    h = length(findobj('type','figure'));
    saveas(figure(h),strcat(savepath,filesep,filename,'.jpeg'))
end

end

function [centres, reward_time, daily_avg_lickhist_plus, daily_avg_lickhist_minus, std_plus, std_minus] = process_lick_data(csplus_t, csminus_t, reward_t, lick_t, edges, t_start, t_end, stimulus_duration, multiplier, n_mice, n_days)
% Process lick data and calculate averages

% Pre-allocate arrays
avg_lickhist_minus = zeros(n_mice, length(edges) - 1);
avg_lickhist_plus = zeros(n_mice, length(edges) - 1);
daily_avg_lickhist_minus = zeros(n_days, length(edges) - 1);
daily_avg_lickhist_plus = zeros(n_days, length(edges) - 1);
std_minus = zeros(n_days, length(edges) - 1);
std_plus = zeros(n_days, length(edges) - 1);

for d = 1:n_days
    for m = 1:n_mice
        plus_trials = length(csplus_t{m}{d});
        minus_trials = length(csminus_t{m}{d}); 
        
        
        % Initialize lick histograms based on the length of edges
        lickhist_plus = zeros(plus_trials, length(edges) - 1);
        lickhist_minus = zeros(minus_trials, length(edges) - 1);

        for trial = 1:plus_trials
            servo_start = csplus_t{m}{d}(trial);
            reward_start = reward_t{m}{d}(trial);
            trial_lick_plus = lick_t{m}{d}((lick_t{m}{d} > ((servo_start+t_start))) & (lick_t{m}{d} < (reward_start+t_end)));          
            trial_lick_plus = trial_lick_plus - servo_start;
            lickhist_plus(trial,:) = histcounts(trial_lick_plus,edges);
        end        

        for trial = 1:minus_trials
            servo_start = csminus_t{m}{d}(trial);
            trial_lick_minus = lick_t{m}{d}((lick_t{m}{d} > ((servo_start+t_start))) & (lick_t{m}{d} < (servo_start+stimulus_duration + t_end)));
            trial_lick_minus = trial_lick_minus - servo_start;
            lickhist_minus(trial,:) = (histcounts(trial_lick_minus,edges));
        end
       
        % average across mice
        avg_lickhist_minus(m,:) = (mean(lickhist_minus)).*multiplier;
        avg_lickhist_plus(m,:) = (mean(lickhist_plus)).*multiplier;
    end

    daily_avg_lickhist_minus(d,:) = mean(avg_lickhist_minus);
    daily_avg_lickhist_plus(d,:) = mean(avg_lickhist_plus);
    std_minus(d,:) = std(avg_lickhist_minus)./sqrt(n_mice);
    std_plus(d,:) = std(avg_lickhist_plus)./sqrt(n_mice);
end

centres = edges(1:end-1)+ diff(edges)/2;
reward_time = (reward_t{m}{d}(1) - csplus_t{m}{d}(1));

end

function create_lick_rate_plots(centres, daily_avg_lickhist_plus, daily_avg_lickhist_minus, std_plus, std_minus, labels, reward_time, stim_in_time, stimulus_duration, t_start, t_end, n_days)
% Create and format the lick rate plots

h = length(findobj('type','figure'));
fh = figure(h+1);
tiledlayout(1, n_days, 'TileSpacing', 'compact', 'Padding', 'compact');

for d = 1:n_days
    nexttile(d)
    shadedErrorBar_cl(centres,daily_avg_lickhist_plus(d,:),std_plus(d,:))
    hold on 
    shadedErrorBar(centres,daily_avg_lickhist_minus(d,:),std_minus(d,:))
    title(labels{d});
    xlabel('Time from trial start (s)', 'FontSize',10)
    ylabel('Lick rate (licks/s)', 'FontSize',10)
    axis([centres(1) centres(end) 0 10])
    xtick_values = t_start:1000:t_end;
    xtick_labels = (xtick_values - t_start) / 1000;
    xticklabels(xtick_labels);
    xticks(xtick_values);
    stim_out = stim_in_time + stimulus_duration;
    xline(reward_time,'--k')
    v = [stim_in_time 0; stim_in_time 10; stim_out 10; stim_out 0];
    f = [1 2 3 4];
    p_l = patch('Faces',f,'Vertices',v,'Facecolor', [0.5 .5 0.5]);
    p_l.FaceAlpha = 0.3;
    p_l.EdgeColor = 'none';
    text(stim_in_time, 9.8, 'Texture', 'FontSize',9)
end

text(6800, 9, 'CS+','Color', [40/255 53/255 192/255])
text(6800, 8.5, 'CS-')
sgtitle('Mean Lick Rate')
fh.Position = [18 455 1501 325];

end
