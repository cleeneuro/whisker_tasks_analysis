%% This is the main code for running lick analysis from textfile outputs fpr Pavlovian tasks
% User must update: whether or not they want to save figures, the path, the
% parent folder for data path and the mousenames you want to analyse 
% Data within this folder structure: textfile_data/animalname/
% filenames must be animalname_date.txt or animal_date_whiskertrim.txt
% day indices are based on # of files in the folder. Files ending in
% _whiskertrim or _whiskertrimmed will be automatically sorted into a whisker
% trim day

close all
clear
clc

%% Configuration
% Set to true to save figures, false to only display them
SAVE_FIGURES = true;

% Base save path - all figures will be saved in subfolders under this path
BASE_SAVE_PATH = '/path/to/your/output_figures';

%% Load data 
% outerpath should contain a subfolder for each mouse named in convention of 'CL001'
% Each subfolder should have text files with the mouse name followed by
% _date i.e. CL001_230801

outerpath = '/path/to/your/textfile_data';

% Specify the mouse names to process
mousenames = {'CL001', 'CL002', 'CL003', 'CL010'};
n_mice = length(mousenames);

%% Extract the data and put into cell arrays. Each mouse has its own cell with sub-cells for each date
% Initialize variables to store file information
mouse_training_files = cell(n_mice, 1);
mouse_whiskertrim_files = cell(n_mice, 1);
all_training_dates = []; % Collect all unique training dates across all mice
max_training_days = 0;

% First pass: collect all files and separate training from whisker trim files
for m = 1:n_mice
    mousefolder = [outerpath, filesep, mousenames{m}];
    
    % Check if mouse folder exists
    if ~exist(mousefolder, 'dir')
        warning('Mouse folder %s does not exist, skipping...', mousenames{m});
        continue;
    end
    
    getfiles = dir(fullfile(mousefolder, '*.txt'));
    mousefiles = getfiles(~ismember({getfiles.name},{'.','..'}));
    
    % Separate training files from whisker trim files
    training_files = {};
    whiskertrim_file = '';
    
    for f = 1:length(mousefiles)
        filename = mousefiles(f).name;
        if contains(filename, '_whiskertrim') || contains(filename, '_whiskertrimmed')
            whiskertrim_file = filename;
        else
            training_files{end+1} = filename;
        end
    end
    
    % Sort training files by date (extract date from filename)
    if ~isempty(training_files)
        file_dates = [];
        for f = 1:length(training_files)
            filename = training_files{f};
            % Extract date from filename (e.g., CL042_20250801.txt or CL042_20250801(1).txt)
            date_match = regexp(filename, '_(\d{8})', 'tokens');
            if ~isempty(date_match)
                file_dates(f) = str2double(date_match{1}{1});
            else
                file_dates(f) = 0; % Default for files without date
            end
        end
        
        % Sort files by date
        [~, sort_idx] = sort(file_dates);
        training_files = training_files(sort_idx);
        
        % Collect all training dates for cross-mouse alignment
        all_training_dates = [all_training_dates, file_dates(sort_idx)];
    end
    
    % Store file information
    mouse_training_files{m} = training_files;
    if ~isempty(whiskertrim_file)
        mouse_whiskertrim_files{m} = whiskertrim_file;
    end
    
    % Track maximum number of training days
    max_training_days = max(max_training_days, length(training_files));
end

% Get unique training dates and sort them to create a master timeline
unique_dates = unique(all_training_dates);
unique_dates = sort(unique_dates(unique_dates > 0)); % Remove 0 values and sort
n_unique_days = length(unique_dates);

% Generate labels automatically based on detected files
labels = {};
for d = 1:n_unique_days
    labels{end+1} = sprintf('Day %d', d);
end

% Add whisker trim label if any mouse has whisker trim data
has_whisker_trim = any(~cellfun(@isempty, mouse_whiskertrim_files));
if has_whisker_trim
    labels{end+1} = 'Whisker Trim';
end

fprintf('Detected %d training days across all mice', n_unique_days);
if has_whisker_trim
    fprintf(' with whisker trim data\n');
else
    fprintf(' without whisker trim data\n');
end

% Second pass: load the actual data with proper day alignment
% Initialize data structures with proper dimensions
for m = 1:n_mice
    csplus_t{m} = cell(1, n_unique_days + 1); % +1 for whisker trim
    csminus_t{m} = cell(1, n_unique_days + 1);
    lick_t{m} = cell(1, n_unique_days + 1);
    reward_t{m} = cell(1, n_unique_days + 1);
    n_days(m) = 0; % Will be updated based on actual data loaded
end

for m = 1:n_mice
    mousefolder = [outerpath, filesep, mousenames{m}];
    
    % Check if mouse folder exists
    if ~exist(mousefolder, 'dir')
        continue;
    end
    
    training_files = mouse_training_files{m};
    whiskertrim_file = mouse_whiskertrim_files{m};
    
    % Load training day data with proper day alignment
    for d = 1:length(training_files)
        cur_day = training_files{d};
        filepath = [mousefolder, filesep, cur_day];
        
        % Extract date from filename to find correct day index
        date_match = regexp(cur_day, '_(\d{8})', 'tokens');
        if ~isempty(date_match)
            file_date = str2double(date_match{1}{1});
            % Find which day index this date corresponds to in the master timeline
            day_idx = find(unique_dates == file_date);
            
            if ~isempty(day_idx)
                t = import_text_file(filepath);
                t = t((~isnan(t.Time)),:);   %remove NaNs
                csplus_t{m}{day_idx} = table2array(t((t.Event == 'CS+'),1));
                csminus_t{m}{day_idx} = table2array(t((t.Event == 'CS-'),1));
                lick_t{m}{day_idx} = table2array(t((t.Event == 'LICK'),1));

                if day_idx == 1 
                    reward_t{m}{day_idx} = csplus_t{m}{day_idx} + 2900;
                else 
                    reward_t{m}{day_idx} = table2array(t((t.Event == 'REWARD'),1));
                end
                
                % Update n_days to track actual days with data
                n_days(m) = max(n_days(m), day_idx);
                
                % Data validation - check for empty arrays
                if isempty(csplus_t{m}{day_idx})
                    fprintf('Warning: No CS+ trials found for mouse %s, day %d (date: %d)\n', mousenames{m}, day_idx, file_date);
                end
                if isempty(csminus_t{m}{day_idx})
                    fprintf('Warning: No CS- trials found for mouse %s, day %d (date: %d)\n', mousenames{m}, day_idx, file_date);
                end
                if isempty(lick_t{m}{day_idx})
                    fprintf('Warning: No lick data found for mouse %s, day %d (date: %d)\n', mousenames{m}, day_idx, file_date);
                end
            else
                fprintf('Warning: Could not align date %d for mouse %s with master timeline\n', file_date, mousenames{m});
            end
        else
            fprintf('Warning: Could not extract date from filename %s for mouse %s\n', cur_day, mousenames{m});
        end
    end
    
    % Load whisker trim data if available (always goes to the last position)
    if ~isempty(whiskertrim_file)
        whiskertrim_path = [mousefolder, filesep, whiskertrim_file];
        t = import_text_file(whiskertrim_path);
        t = t((~isnan(t.Time)),:);   %remove NaNs
        csplus_t{m}{n_unique_days + 1} = table2array(t((t.Event == 'CS+'),1));
        csminus_t{m}{n_unique_days + 1} = table2array(t((t.Event == 'CS-'),1));
        lick_t{m}{n_unique_days + 1} = table2array(t((t.Event == 'LICK'),1));
        reward_t{m}{n_unique_days + 1} = table2array(t((t.Event == 'REWARD'),1));
        
        % Data validation for whisker trim
        if isempty(csplus_t{m}{n_unique_days + 1})
            fprintf('Warning: No CS+ trials found for mouse %s, whisker trim\n', mousenames{m});
        end
        if isempty(csminus_t{m}{n_unique_days + 1})
            fprintf('Warning: No CS- trials found for mouse %s, whisker trim\n', mousenames{m});
        end
        if isempty(lick_t{m}{n_unique_days + 1})
            fprintf('Warning: No lick data found for mouse %s, whisker trim\n', mousenames{m});
        end
    end
end

%% Check if any data was loaded
if ~exist('csplus_t', 'var') || isempty(csplus_t)
    fprintf('Data loading failed. Checking paths...\n');
    fprintf('Outer path: %s\n', outerpath);
    fprintf('Mouse names: %s\n', strjoin(mousenames, ', '));
    for m = 1:n_mice
        mousefolder = [outerpath, filesep, mousenames{m}];
        fprintf('Checking mouse folder: %s\n', mousefolder);
        if exist(mousefolder, 'dir')
            fprintf('  -> Folder exists\n');
        else
            fprintf('  -> Folder does NOT exist\n');
        end
    end
    error('No data was loaded. Please check that the mouse folders exist in the specified path: %s', outerpath);
end

%% Create save paths for different analysis types
if SAVE_FIGURES
    % Create subfolders for different analysis types
    raster_savepath = fullfile(BASE_SAVE_PATH, 'rasterplots');
    lickrate_savepath = fullfile(BASE_SAVE_PATH, 'lickrate_plots');
    histogram_savepath = fullfile(BASE_SAVE_PATH, 'lick_histograms');
    
    % Create directories if they don't exist
    if ~exist(raster_savepath, 'dir')
        mkdir(raster_savepath);
    end
    if ~exist(lickrate_savepath, 'dir')
        mkdir(lickrate_savepath);
    end
    if ~exist(histogram_savepath, 'dir')
        mkdir(histogram_savepath);
    end
else
    % Set empty save paths when not saving
    raster_savepath = '';
    lickrate_savepath = '';
    histogram_savepath = '';
end

%% Make Licking Raster Plots 
if ~isempty(raster_savepath)
    lickrasterplot(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels, raster_savepath);
else
    lickrasterplot(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels);
end

%% Lick rate on all days, averaged across mice 
if ~isempty(lickrate_savepath)
    lick_rate_plot(csplus_t, csminus_t, reward_t, lick_t, labels, lickrate_savepath)
else
    lick_rate_plot(csplus_t, csminus_t, reward_t, lick_t, labels)
end

%% Lick rate, one mouse all days
if ~isempty(lickrate_savepath)
   lick_rate_plot_mouse_all_days(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels, lickrate_savepath)
else
   lick_rate_plot_mouse_all_days(csplus_t, csminus_t, reward_t, lick_t, mousenames, labels)
end

%% Histogram of anticipatory lick count for CS plus and CS minus 
number_licks_histogram(csplus_t, csminus_t, reward_t, lick_t, histogram_savepath, labels);
