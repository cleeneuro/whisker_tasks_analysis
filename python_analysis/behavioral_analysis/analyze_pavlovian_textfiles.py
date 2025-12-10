"""
Behavioural analysis of Pavlovian conditioning data using textfile outputs. 
Requires textfile outputs from PavlovianTexture or GradedPavlovianTexture arduino code.

User must enter the mouse names to analyze. 
The textfiles must be located in a folder with the mouse name

User should set DATA_DIRECTORY and BASE_OUTPUT_FOLDER

Usage: 
python analyze_textpav_textfiles.py
User will be prompted to select which analyses to run.
Output will be saved to BASE_OUTPUT_FOLDER.

"""
#%% CONFIGURATION AND IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Analysis Constants
SERVO_MOVE_DUR = 1120  # Servo move duration in milliseconds
STIM_PRESENT_DUR = 2000  # Stimulus present duration in milliseconds
ANT_LICK_THRESHOLD = 3  # Threshold for anticipatory licking

# Lick Rate Analysis Constants
LICK_RATE_BIN_SIZE = 200  # Bin size in milliseconds
LICK_RATE_PRE_TIME = 3000  # Time before CS onset to analyze (ms)
LICK_RATE_POST_TIME = 8000  # Time after CS onset to analyze (ms)
#REWARD_TIME_OFFSET = SERVO_MOVE_DUR + STIM_PRESENT_DUR  # When reward is delivered
REWARD_TIME_OFFSET = 2700

# File Paths
DATA_DIRECTORY = '/path/to/your/textfile_data/'
BASE_OUTPUT_FOLDER = '/path/to/your/output_figures/'

# Mouse Selection - Specify which mice to analyze
MOUSE_FOLDERS = ['mousename1', 'mousename2', 'mousename3']  # Set to None to analyze all folders
# Create output folder name based on selected mice
if MOUSE_FOLDERS is not None:
    mouse_names_str = '_'.join(sorted(MOUSE_FOLDERS))
    OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, mouse_names_str)
else:
    OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'all_mice')


#%% CORE DATA PROCESSING FUNCTIONS

def get_mouse_filepaths(directory, folder_names=None):
    """
    Get file paths for specified mouse folders, separating regular training files from whisker trim files.
    
    Args:
        directory: Path to directory containing mouse folders
        folder_names: List of specific folder names to analyze (None = all folders)
    
    Returns:
        DataFrame of training file paths, dict of whisker trim file paths, list of mouse names
    """
    print("=" * 60)
    print("LOADING DATA FILES")
    print("=" * 60)
    
    mouse_training_filepaths = {}
    mouse_whiskertrim_filepaths = {}
    
    if folder_names is None:
        mouse_names = os.listdir(directory)
        print(f"Analyzing ALL mouse folders in: {directory}")
    else:
        mouse_names = folder_names
        print(f"Analyzing SELECTED mouse folders: {folder_names}")
    
    for mouse_name in mouse_names:
        subfolder_path = os.path.join(directory, mouse_name)
        if os.path.isdir(subfolder_path):
            all_files = [
                os.path.join(subfolder_path, f) 
                for f in os.listdir(subfolder_path) 
                if os.path.isfile(os.path.join(subfolder_path, f)) and '@' not in f
            ]
            
            # Separate regular training files from whisker trim files
            training_files = []
            whiskertrim_file = None
            
            for filepath in sorted(all_files):
                if '_whiskertrim.txt' in filepath:
                    whiskertrim_file = filepath
                else:
                    training_files.append(filepath)
            
            mouse_training_filepaths[mouse_name] = training_files
            if whiskertrim_file:
                mouse_whiskertrim_filepaths[mouse_name] = whiskertrim_file
            
            whisker_info = f" (+ whisker trim)" if whiskertrim_file else ""
            print(f"{mouse_name}: Found {len(training_files)} training files{whisker_info}")
        else:
            print(f"{mouse_name}: Folder not found!")
    
    if not mouse_training_filepaths:
        print("ERROR: No valid mouse folders found!")
        return pd.DataFrame(), {}, []
    
    # Pad shorter training file lists with None values
    max_training_days = max([len(v) for v in mouse_training_filepaths.values()])
    for key in mouse_training_filepaths:
        mouse_training_filepaths[key].extend([None] * (max_training_days - len(mouse_training_filepaths[key])))
    
    print(f"Maximum training days: {max_training_days}")
    print(f"Mice with whisker trim data: {len(mouse_whiskertrim_filepaths)}")
    return pd.DataFrame(mouse_training_filepaths), mouse_whiskertrim_filepaths, list(mouse_training_filepaths.keys())

def load_behavioral_data(filepath):
    """
    Load and parse behavioral data from a text file.
    
    Args:
        filepath: Path to text file
    
    Returns:
        DataFrame with Time and Event columns
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
    
    data = []
    for line in lines:
        line = line.strip()
        if "Lick count" in line or "Timeout" in line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                time = float(parts[0])
                event = ' '.join(parts[1:])
                data.append([time, event])
            except ValueError:
                continue
    
    df = pd.DataFrame(data, columns=["Time", "Event"])
    df["Event"] = df["Event"].astype("category")
    return df

#%% ANALYSIS 1: LICK INDEX CALCULATION

def calculate_lick_index(stim_times, lick_times):
    """
    Calculate lick index: (stim_licks - baseline_licks) / (stim_licks + baseline_licks)
    
    Args:
        stim_times: Array of stimulus onset times
        lick_times: Array of lick times
    
    Returns:
        Array of lick indices for each trial
    """
    lick_indices = []
    for stim_time in stim_times:
        stim_in_time = stim_time + SERVO_MOVE_DUR
        stim_out_time = stim_in_time + STIM_PRESENT_DUR
        baseline_start_time = stim_in_time - 2000
        baseline_end_time = stim_in_time
        
        stim_lick_indices = np.where((lick_times > stim_in_time) & (lick_times < stim_out_time))[0]
        baseline_lick_indices = np.where((lick_times > baseline_start_time) & (lick_times < baseline_end_time))[0]
        
        stim_lick_count = len(stim_lick_indices)
        baseline_lick_count = len(baseline_lick_indices)
        
        if stim_lick_count == 0 and baseline_lick_count == 0:
            lick_index = 0
        else:
            lick_index = (stim_lick_count - baseline_lick_count) / (stim_lick_count + baseline_lick_count)
        
        lick_indices.append(lick_index)
    
    return np.array(lick_indices)

def run_lick_index_analysis(filepaths_df, whiskertrim_filepaths, mouse_names):
    """
    Run lick index analysis for all mice and days, including whisker trim data.
    
    Args:
        filepaths_df: DataFrame with training day filepaths
        whiskertrim_filepaths: Dict with whisker trim filepaths per mouse
        mouse_names: List of mouse names
    
    Returns:
        Dictionary with lick index results including whisker trim data
    """
    print("=" * 60)
    print("RUNNING LICK INDEX ANALYSIS")
    print("=" * 60)
    
    n_training_days = filepaths_df.shape[0]
    n_mice = len(mouse_names)
    
    # Training days data
    CSplus_lick_index = np.full((n_training_days, n_mice), np.nan)
    CSminus_lick_index = np.full((n_training_days, n_mice), np.nan)
    
    # Whisker trim data (single day)
    CSplus_whiskertrim = np.full(n_mice, np.nan)
    CSminus_whiskertrim = np.full(n_mice, np.nan)
    
    for mouse_idx, mouse_name in enumerate(mouse_names):
        print(f"\nProcessing mouse: {mouse_name}")
        
        # Process training days
        for day_idx in range(n_training_days):
            filepath = filepaths_df.loc[day_idx, mouse_name]
            
            if filepath is None or pd.isna(filepath):
                print(f"  Day {day_idx + 1}: No file (skipped)")
                continue
            
            print(f"  Day {day_idx + 1}: Processing...")
            events_df = load_behavioral_data(filepath)
            lick_times = np.array(events_df[events_df['Event'] == 'LICK']['Time'].values)
            cs_plus_times = np.array(events_df[events_df['Event'] == 'CS+']['Time'].values)
            cs_minus_times = np.array(events_df[events_df['Event'] == 'CS-']['Time'].values)
            
            CSplus_lick_index[day_idx, mouse_idx] = np.mean(calculate_lick_index(cs_plus_times, lick_times))
            CSminus_lick_index[day_idx, mouse_idx] = np.mean(calculate_lick_index(cs_minus_times, lick_times))
        
        # Process whisker trim data if available
        if mouse_name in whiskertrim_filepaths:
            print(f"  Whisker trim: Processing...")
            events_df = load_behavioral_data(whiskertrim_filepaths[mouse_name])
            lick_times = np.array(events_df[events_df['Event'] == 'LICK']['Time'].values)
            cs_plus_times = np.array(events_df[events_df['Event'] == 'CS+']['Time'].values)
            cs_minus_times = np.array(events_df[events_df['Event'] == 'CS-']['Time'].values)
            
            CSplus_whiskertrim[mouse_idx] = np.mean(calculate_lick_index(cs_plus_times, lick_times))
            CSminus_whiskertrim[mouse_idx] = np.mean(calculate_lick_index(cs_minus_times, lick_times))
    
    # Calculate averages for training days
    CSplus_avg = np.nanmean(CSplus_lick_index, axis=1)
    CSminus_avg = np.nanmean(CSminus_lick_index, axis=1)
    
    # Calculate averages for whisker trim
    CSplus_whiskertrim_avg = np.nanmean(CSplus_whiskertrim)
    CSminus_whiskertrim_avg = np.nanmean(CSminus_whiskertrim)
    
    print(f"\nLick Index Analysis Complete!")
    return {
        'CSplus_lick_index': CSplus_lick_index,
        'CSminus_lick_index': CSminus_lick_index,
        'CSplus_avg': CSplus_avg,
        'CSminus_avg': CSminus_avg,
        'CSplus_whiskertrim': CSplus_whiskertrim,
        'CSminus_whiskertrim': CSminus_whiskertrim,
        'CSplus_whiskertrim_avg': CSplus_whiskertrim_avg,
        'CSminus_whiskertrim_avg': CSminus_whiskertrim_avg,
        'mouse_names': mouse_names
    }

#%% ANALYSIS 2: ANTICIPATORY LICKING PROBABILITY

def calculate_anticipatory_licking_probability(stim_times, lick_times):
    """
    Calculate probability of anticipatory licking (>3 licks during stimulus).
    
    Args:
        stim_times: Array of stimulus onset times
        lick_times: Array of lick times
    
    Returns:
        Probability of anticipatory licking
    """
    n_trials = len(stim_times)
    trials_with_ant_licks = 0
    
    for stim_time in stim_times:
        stim_in_time = stim_time + SERVO_MOVE_DUR
        stim_out_time = stim_in_time + STIM_PRESENT_DUR
        ant_licks = np.where((lick_times > stim_in_time) & (lick_times < stim_out_time))[0]
        if len(ant_licks) > ANT_LICK_THRESHOLD:
            trials_with_ant_licks += 1
    
    return trials_with_ant_licks / n_trials

def run_anticipatory_licking_analysis(filepaths_df, whiskertrim_filepaths, mouse_names):
    """
    Run anticipatory licking analysis for all mice and days, including whisker trim data.
    
    Args:
        filepaths_df: DataFrame with training day filepaths
        whiskertrim_filepaths: Dict with whisker trim filepaths per mouse
        mouse_names: List of mouse names
    
    Returns:
        Dictionary with anticipatory licking results including whisker trim data
    """
    print("=" * 60)
    print("RUNNING ANTICIPATORY LICKING ANALYSIS")
    print("=" * 60)
    
    n_training_days = filepaths_df.shape[0]
    n_mice = len(mouse_names)
    
    # Training days data
    p_ant_licks_plus = np.full((n_training_days, n_mice), np.nan)
    p_ant_licks_minus = np.full((n_training_days, n_mice), np.nan)
    
    # Whisker trim data (single day)
    p_ant_whiskertrim_plus = np.full(n_mice, np.nan)
    p_ant_whiskertrim_minus = np.full(n_mice, np.nan)
    
    for mouse_idx, mouse_name in enumerate(mouse_names):
        print(f"\nProcessing mouse: {mouse_name}")
        
        # Process training days
        for day_idx in range(n_training_days):
            filepath = filepaths_df.loc[day_idx, mouse_name]
            
            if filepath is None or pd.isna(filepath):
                print(f"  Day {day_idx + 1}: No file (skipped)")
                continue
            
            print(f"  Day {day_idx + 1}: Processing...")
            events_df = load_behavioral_data(filepath)
            lick_times = np.array(events_df[events_df['Event'] == 'LICK']['Time'].values)
            cs_plus_times = np.array(events_df[events_df['Event'] == 'CS+']['Time'].values)
            cs_minus_times = np.array(events_df[events_df['Event'] == 'CS-']['Time'].values)
            
            p_ant_licks_plus[day_idx, mouse_idx] = calculate_anticipatory_licking_probability(cs_plus_times, lick_times)
            p_ant_licks_minus[day_idx, mouse_idx] = calculate_anticipatory_licking_probability(cs_minus_times, lick_times)
        
        # Process whisker trim data if available
        if mouse_name in whiskertrim_filepaths:
            print(f"  Whisker trim: Processing...")
            events_df = load_behavioral_data(whiskertrim_filepaths[mouse_name])
            lick_times = np.array(events_df[events_df['Event'] == 'LICK']['Time'].values)
            cs_plus_times = np.array(events_df[events_df['Event'] == 'CS+']['Time'].values)
            cs_minus_times = np.array(events_df[events_df['Event'] == 'CS-']['Time'].values)
            
            p_ant_whiskertrim_plus[mouse_idx] = calculate_anticipatory_licking_probability(cs_plus_times, lick_times)
            p_ant_whiskertrim_minus[mouse_idx] = calculate_anticipatory_licking_probability(cs_minus_times, lick_times)
        
        # Individual mouse data is processed but not saved to .txt files
    
    # Calculate averages for training days
    p_ant_plus_avg = np.nanmean(p_ant_licks_plus, axis=1)
    p_ant_minus_avg = np.nanmean(p_ant_licks_minus, axis=1)
    
    # Calculate averages for whisker trim
    p_ant_whiskertrim_plus_avg = np.nanmean(p_ant_whiskertrim_plus)
    p_ant_whiskertrim_minus_avg = np.nanmean(p_ant_whiskertrim_minus)
    
    print(f"\nAnticipatory Licking Analysis Complete!")
    return {
        'p_ant_licks_plus': p_ant_licks_plus,
        'p_ant_licks_minus': p_ant_licks_minus,
        'p_ant_plus_avg': p_ant_plus_avg,
        'p_ant_minus_avg': p_ant_minus_avg,
        'p_ant_whiskertrim_plus': p_ant_whiskertrim_plus,
        'p_ant_whiskertrim_minus': p_ant_whiskertrim_minus,
        'p_ant_whiskertrim_plus_avg': p_ant_whiskertrim_plus_avg,
        'p_ant_whiskertrim_minus_avg': p_ant_whiskertrim_minus_avg,
        'mouse_names': mouse_names
    }

#%% ANALYSIS 3: LICK RATE OVER TIME

def calculate_lick_rate_over_time(stim_times, lick_times):
    """
    Calculate lick rate over time for a set of trials.
    
    Args:
        stim_times: Array of stimulus onset times
        lick_times: Array of lick times
    
    Returns:
        time_centers: Array of time points (relative to stimulus onset)
        lick_rates_per_trial: Array of lick rates in Hz for each trial and time bin
        lick_rates_avg: Array of average lick rates in Hz across trials
    """
    # Create time bins relative to stimulus onset
    time_start = -LICK_RATE_PRE_TIME
    time_end = LICK_RATE_POST_TIME
    time_bins = np.arange(time_start, time_end + LICK_RATE_BIN_SIZE, LICK_RATE_BIN_SIZE)
    time_centers = time_bins[:-1] + LICK_RATE_BIN_SIZE / 2
    
    # Initialize array to store lick counts for each trial and time bin
    lick_counts_per_trial = np.zeros((len(stim_times), len(time_bins) - 1))
    
    for trial_idx, stim_time in enumerate(stim_times):
        # Get licks relative to this stimulus onset
        relative_lick_times = lick_times - stim_time
        
        # Count licks in each time bin
        for bin_idx, (bin_start, bin_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            licks_in_bin = np.sum((relative_lick_times >= bin_start) & (relative_lick_times < bin_end))
            lick_counts_per_trial[trial_idx, bin_idx] = licks_in_bin
    
    # Convert to lick rates in Hz
    bin_duration_sec = LICK_RATE_BIN_SIZE / 1000.0  # Convert ms to seconds
    lick_rates_per_trial = lick_counts_per_trial / bin_duration_sec
    
    # Calculate average lick rate across trials
    lick_rates_avg = np.mean(lick_rates_per_trial, axis=0)
    
    return time_centers, lick_rates_per_trial, lick_rates_avg

def run_lick_rate_analysis(filepaths_df, mouse_names):
    """
    Run lick rate over time analysis for all mice and days.
    
    Returns:
        Dictionary with lick rate results organized by day
    """
    print("=" * 60)
    print("RUNNING LICK RATE OVER TIME ANALYSIS")
    print("=" * 60)
    
    n_days = filepaths_df.shape[0]
    n_mice = len(mouse_names)
    
    # Store results by day
    results_by_day = {}
    
    for day_idx in range(n_days):
        print(f"\nProcessing Day {day_idx + 1}:")
        
        # Initialize storage for this day - track which mice have data
        cs_plus_rates_by_mouse = {}
        cs_minus_rates_by_mouse = {}
        time_centers = None
        mice_with_data = []
        
        for mouse_idx, mouse_name in enumerate(mouse_names):
            filepath = filepaths_df.loc[day_idx, mouse_name]
            
            if filepath is None or pd.isna(filepath):
                print(f"  {mouse_name}: No file (skipped)")
                continue
            
            print(f"  {mouse_name}: Processing...")
            events_df = load_behavioral_data(filepath)
            lick_times = np.array(events_df[events_df['Event'] == 'LICK']['Time'].values)
            cs_plus_times = np.array(events_df[events_df['Event'] == 'CS+']['Time'].values)
            cs_minus_times = np.array(events_df[events_df['Event'] == 'CS-']['Time'].values)
            
            if len(cs_plus_times) > 0:
                time_centers, cs_plus_rates_per_trial, cs_plus_rates_avg = calculate_lick_rate_over_time(cs_plus_times, lick_times)
                cs_plus_rates_by_mouse[mouse_name] = {
                    'per_trial': cs_plus_rates_per_trial,
                    'avg': cs_plus_rates_avg
                }
                mice_with_data.append(mouse_name)
            
            if len(cs_minus_times) > 0:
                time_centers, cs_minus_rates_per_trial, cs_minus_rates_avg = calculate_lick_rate_over_time(cs_minus_times, lick_times)
                cs_minus_rates_by_mouse[mouse_name] = {
                    'per_trial': cs_minus_rates_per_trial,
                    'avg': cs_minus_rates_avg
                }
        
        # Store results for this day
        if cs_plus_rates_by_mouse and time_centers is not None:
            results_by_day[day_idx] = {
                'time_centers': time_centers,
                'cs_plus_rates_by_mouse': cs_plus_rates_by_mouse,
                'cs_minus_rates_by_mouse': cs_minus_rates_by_mouse,
                'mice_with_data': mice_with_data,
                'day': day_idx + 1
            }
    
    print(f"\nLick Rate Analysis Complete for {len(results_by_day)} days!")
    return results_by_day

#%% PLOTTING FUNCTIONS

def plot_lick_index_results(results, n_training_days):
    """
    Create matplotlib plot for lick index results including whisker trim data (SSH compatible).
    """
    print("=" * 60)
    print("CREATING LICK INDEX PLOT")
    print("=" * 60)
    
    CSplus_data = results['CSplus_lick_index']
    CSminus_data = results['CSminus_lick_index']
    CSplus_whiskertrim = results['CSplus_whiskertrim']
    CSminus_whiskertrim = results['CSminus_whiskertrim']
    
    # Calculate SEM with NaN handling for training days
    valid_counts_plus = np.sum(~np.isnan(CSplus_data), axis=1)
    valid_counts_minus = np.sum(~np.isnan(CSminus_data), axis=1)
    
    CSplus_sem = np.where(valid_counts_plus > 0, 
                          np.nanstd(CSplus_data, axis=1) / np.sqrt(valid_counts_plus), 0)
    CSminus_sem = np.where(valid_counts_minus > 0, 
                           np.nanstd(CSminus_data, axis=1) / np.sqrt(valid_counts_minus), 0)
    
    # Calculate SEM for whisker trim data
    valid_counts_plus_wt = np.sum(~np.isnan(CSplus_whiskertrim))
    valid_counts_minus_wt = np.sum(~np.isnan(CSminus_whiskertrim))
    
    CSplus_whiskertrim_sem = np.nanstd(CSplus_whiskertrim) / np.sqrt(valid_counts_plus_wt) if valid_counts_plus_wt > 0 else 0
    CSminus_whiskertrim_sem = np.nanstd(CSminus_whiskertrim) / np.sqrt(valid_counts_minus_wt) if valid_counts_minus_wt > 0 else 0
    
    # Check if we have any whisker trim data
    has_whisker_trim_data = valid_counts_plus_wt > 0 or valid_counts_minus_wt > 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Training days x-axis
    training_days_x = range(1, n_training_days + 1)
    
    # Plot individual mouse training data (faint lines)
    for i in range(CSplus_data.shape[1]):
        # Find the last valid training day for this mouse
        cs_plus_mouse_data = CSplus_data[:, i]
        cs_minus_mouse_data = CSminus_data[:, i]
        
        # Find last non-NaN index
        valid_plus_indices = ~np.isnan(cs_plus_mouse_data)
        valid_minus_indices = ~np.isnan(cs_minus_mouse_data)
        
        if np.any(valid_plus_indices):
            last_valid_plus = np.where(valid_plus_indices)[0][-1] + 1
            ax.plot(range(1, last_valid_plus + 1), cs_plus_mouse_data[:last_valid_plus], 'b-', alpha=0.3, linewidth=1)
        
        if np.any(valid_minus_indices):
            last_valid_minus = np.where(valid_minus_indices)[0][-1] + 1
            ax.plot(range(1, last_valid_minus + 1), cs_minus_mouse_data[:last_valid_minus], 'k-', alpha=0.3, linewidth=1)
        
        # Add whisker trim data for this mouse if available (only if we have whisker trim data overall)
        if has_whisker_trim_data:
            if not np.isnan(CSplus_whiskertrim[i]):
                # Connect last training day to whisker trim with a faint line
                if np.any(valid_plus_indices):
                    last_training_value = cs_plus_mouse_data[last_valid_plus - 1]
                    ax.plot([last_valid_plus, n_training_days + 2], [last_training_value, CSplus_whiskertrim[i]], 'b--', alpha=0.2, linewidth=1)
                ax.plot(n_training_days + 2, CSplus_whiskertrim[i], 'bo', alpha=0.3, markersize=4)
            
            if not np.isnan(CSminus_whiskertrim[i]):
                # Connect last training day to whisker trim with a faint line
                if np.any(valid_minus_indices):
                    last_training_value = cs_minus_mouse_data[last_valid_minus - 1]
                    ax.plot([last_valid_minus, n_training_days + 2], [last_training_value, CSminus_whiskertrim[i]], 'k--', alpha=0.2, linewidth=1)
                ax.plot(n_training_days + 2, CSminus_whiskertrim[i], 'ko', alpha=0.3, markersize=4)
    
    # Plot training day averages with error bars
    ax.errorbar(training_days_x, results['CSplus_avg'], yerr=CSplus_sem,
                label='CS+ Lick Index', color='blue', fmt='-o', linewidth=2, capsize=5)
    ax.errorbar(training_days_x, results['CSminus_avg'], yerr=CSminus_sem,
                label='CS- Lick Index', color='black', fmt='-o', linewidth=2, capsize=5)
    
    # Only include whisker trim section if there's whisker trim data
    if has_whisker_trim_data:
        # Plot whisker trim averages with error bars
        if valid_counts_plus_wt > 0:
            ax.errorbar(n_training_days + 2, results['CSplus_whiskertrim_avg'], yerr=CSplus_whiskertrim_sem,
                        color='blue', fmt='o', linewidth=2, capsize=5, markersize=8)
        
        if valid_counts_minus_wt > 0:
            ax.errorbar(n_training_days + 2, results['CSminus_whiskertrim_avg'], yerr=CSminus_whiskertrim_sem,
                        color='black', fmt='o', linewidth=2, capsize=5, markersize=8)
        
        # Customize x-axis with whisker trim
        x_ticks = list(training_days_x) + [n_training_days + 2]
        x_labels = [str(i) for i in training_days_x] + ['Whisker\nTrim']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_title('Lick Index Over Training Days + Whisker Trim')
    else:
        # Customize x-axis without whisker trim
        ax.set_xticks(training_days_x)
        ax.set_xticklabels([str(i) for i in training_days_x])
        ax.set_title('Lick Index Over Training Days')
    
    ax.set_xlabel('Training Day')
    ax.set_ylabel('Lick Index')
    ax.legend()
    ax.grid(False)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plot_file = f'{OUTPUT_FOLDER}/lick_index_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()  # Display interactive plot window
    print(f"Plot saved to: {plot_file}")
    
    plt.close()  # Close figure to free memory

def plot_anticipatory_licking_results(results, n_training_days):
    """
    Create matplotlib plot for anticipatory licking results including whisker trim data (SSH compatible).
    """
    print("=" * 60)
    print("CREATING ANTICIPATORY LICKING PLOT")
    print("=" * 60)
    
    p_ant_plus_data = results['p_ant_licks_plus']
    p_ant_minus_data = results['p_ant_licks_minus']
    p_ant_whiskertrim_plus = results['p_ant_whiskertrim_plus']
    p_ant_whiskertrim_minus = results['p_ant_whiskertrim_minus']
    
    # Calculate SEM with NaN handling for training days
    valid_counts_plus = np.sum(~np.isnan(p_ant_plus_data), axis=1)
    valid_counts_minus = np.sum(~np.isnan(p_ant_minus_data), axis=1)
    
    p_ant_plus_sem = np.where(valid_counts_plus > 0,
                              np.nanstd(p_ant_plus_data, axis=1) / np.sqrt(valid_counts_plus), 0)
    p_ant_minus_sem = np.where(valid_counts_minus > 0,
                               np.nanstd(p_ant_minus_data, axis=1) / np.sqrt(valid_counts_minus), 0)
    
    # Calculate SEM for whisker trim data
    valid_counts_plus_wt = np.sum(~np.isnan(p_ant_whiskertrim_plus))
    valid_counts_minus_wt = np.sum(~np.isnan(p_ant_whiskertrim_minus))
    
    p_ant_whiskertrim_plus_sem = np.nanstd(p_ant_whiskertrim_plus) / np.sqrt(valid_counts_plus_wt) if valid_counts_plus_wt > 0 else 0
    p_ant_whiskertrim_minus_sem = np.nanstd(p_ant_whiskertrim_minus) / np.sqrt(valid_counts_minus_wt) if valid_counts_minus_wt > 0 else 0
    
    # Check if we have any whisker trim data
    has_whisker_trim_data = valid_counts_plus_wt > 0 or valid_counts_minus_wt > 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Training days x-axis
    training_days_x = range(1, n_training_days + 1)
    
    # Plot individual mouse training data (faint lines)
    for i in range(p_ant_plus_data.shape[1]):
        # Find the last valid training day for this mouse
        p_ant_plus_mouse_data = p_ant_plus_data[:, i]
        p_ant_minus_mouse_data = p_ant_minus_data[:, i]
        
        # Find last non-NaN index
        valid_plus_indices = ~np.isnan(p_ant_plus_mouse_data)
        valid_minus_indices = ~np.isnan(p_ant_minus_mouse_data)
        
        if np.any(valid_plus_indices):
            last_valid_plus = np.where(valid_plus_indices)[0][-1] + 1
            ax.plot(range(1, last_valid_plus + 1), p_ant_plus_mouse_data[:last_valid_plus], 'b-', alpha=0.3, linewidth=1)
        
        if np.any(valid_minus_indices):
            last_valid_minus = np.where(valid_minus_indices)[0][-1] + 1
            ax.plot(range(1, last_valid_minus + 1), p_ant_minus_mouse_data[:last_valid_minus], 'k-', alpha=0.3, linewidth=1)
        
        # Add whisker trim data for this mouse if available (only if we have whisker trim data overall)
        if has_whisker_trim_data:
            if not np.isnan(p_ant_whiskertrim_plus[i]):
                # Connect last training day to whisker trim with a faint line
                if np.any(valid_plus_indices):
                    last_training_value = p_ant_plus_mouse_data[last_valid_plus - 1]
                    ax.plot([last_valid_plus, n_training_days + 2], [last_training_value, p_ant_whiskertrim_plus[i]], 'b--', alpha=0.2, linewidth=1)
                ax.plot(n_training_days + 2, p_ant_whiskertrim_plus[i], 'bo', alpha=0.3, markersize=4)
            
            if not np.isnan(p_ant_whiskertrim_minus[i]):
                # Connect last training day to whisker trim with a faint line
                if np.any(valid_minus_indices):
                    last_training_value = p_ant_minus_mouse_data[last_valid_minus - 1]
                    ax.plot([last_valid_minus, n_training_days + 2], [last_training_value, p_ant_whiskertrim_minus[i]], 'k--', alpha=0.2, linewidth=1)
                ax.plot(n_training_days + 2, p_ant_whiskertrim_minus[i], 'ko', alpha=0.3, markersize=4)
    
    # Plot training day averages with error bars
    ax.errorbar(training_days_x, results['p_ant_plus_avg'], yerr=p_ant_plus_sem,
                label='CS+ Anticipatory Licking', color='blue', fmt='-o', linewidth=2, capsize=5)
    ax.errorbar(training_days_x, results['p_ant_minus_avg'], yerr=p_ant_minus_sem,
                label='CS- Anticipatory Licking', color='black', fmt='-o', linewidth=2, capsize=5)
    
    # Only include whisker trim section if there's whisker trim data
    if has_whisker_trim_data:
        # Plot whisker trim averages with error bars
        if valid_counts_plus_wt > 0:
            ax.errorbar(n_training_days + 2, results['p_ant_whiskertrim_plus_avg'], yerr=p_ant_whiskertrim_plus_sem,
                        color='blue', fmt='o', linewidth=2, capsize=5, markersize=8)
        
        if valid_counts_minus_wt > 0:
            ax.errorbar(n_training_days + 2, results['p_ant_whiskertrim_minus_avg'], yerr=p_ant_whiskertrim_minus_sem,
                        color='black', fmt='o', linewidth=2, capsize=5, markersize=8)
        
        # Customize x-axis with whisker trim
        x_ticks = list(training_days_x) + [n_training_days + 2]
        x_labels = [str(i) for i in training_days_x] + ['Whisker\nTrim']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_title('Anticipatory Licking Probability Over Training Days + Whisker Trim')
    else:
        # Customize x-axis without whisker trim
        ax.set_xticks(training_days_x)
        ax.set_xticklabels([str(i) for i in training_days_x])
        ax.set_title('Anticipatory Licking Probability Over Training Days')
    
    ax.set_xlabel('Training Day') 
    ax.set_ylabel(f'Probability of Anticipatory Licking (> {ANT_LICK_THRESHOLD})')
    ax.legend()
    ax.grid(False)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    plot_file = f'{OUTPUT_FOLDER}/anticipatory_lick_probability_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()  # Display interactive plot window
    print(f"Plot saved to: {plot_file}")
    
    plt.close()  # Close figure to free memory

def plot_lick_rate_avg(results_by_day):
    """
    Create lick rate over time plots - average across mice
    """
    print("=" * 60)
    print("CREATING LICK RATE OVER TIME PLOTS")
    print("=" * 60)
    
    for day_idx, results in results_by_day.items():
        print(f"Creating plot for Day {results['day']}...")
        
        time_centers = results['time_centers'] / 1000.0  # Convert to seconds
        cs_plus_rates_by_mouse = results['cs_plus_rates_by_mouse']
        cs_minus_rates_by_mouse = results['cs_minus_rates_by_mouse']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert dictionary data to arrays for plotting
        cs_plus_rates_list = []
        cs_minus_rates_list = []
        
        for mouse_name in cs_plus_rates_by_mouse.keys():
            cs_plus_rates_list.append(cs_plus_rates_by_mouse[mouse_name]['avg'])
            
        for mouse_name in cs_minus_rates_by_mouse.keys():
            cs_minus_rates_list.append(cs_minus_rates_by_mouse[mouse_name]['avg'])
        
        if cs_plus_rates_list:
            cs_plus_rates = np.array(cs_plus_rates_list)
            
            # Plot individual mouse data (thin transparent lines)
            for i in range(cs_plus_rates.shape[0]):
                ax.plot(time_centers, cs_plus_rates[i, :], 'b-', alpha=0.3, linewidth=1)
            
            # Calculate averages and SEM
            cs_plus_avg = np.mean(cs_plus_rates, axis=0)
            cs_plus_sem = np.std(cs_plus_rates, axis=0) / np.sqrt(cs_plus_rates.shape[0])
            
            # Plot averages with shaded SEM
            ax.plot(time_centers, cs_plus_avg, 'b-', linewidth=3, label='CS+ Average')
            ax.fill_between(time_centers, cs_plus_avg - cs_plus_sem, cs_plus_avg + cs_plus_sem, 
                            color='blue', alpha=0.2)
        
        if cs_minus_rates_list:
            cs_minus_rates = np.array(cs_minus_rates_list)
            
            # Plot individual mouse data (thin transparent lines)
            for i in range(cs_minus_rates.shape[0]):
                ax.plot(time_centers, cs_minus_rates[i, :], 'k-', alpha=0.3, linewidth=1)
            
            # Calculate averages and SEM
            cs_minus_avg = np.mean(cs_minus_rates, axis=0)
            cs_minus_sem = np.std(cs_minus_rates, axis=0) / np.sqrt(cs_minus_rates.shape[0])
            
            # Plot averages with shaded SEM
            ax.plot(time_centers, cs_minus_avg, 'k-', linewidth=3, label='CS- Average')
            ax.fill_between(time_centers, cs_minus_avg - cs_minus_sem, cs_minus_avg + cs_minus_sem, 
                            color='black', alpha=0.2)
        
        # Add simple grey patch from SERVO_MOVE_DUR to SERVO_MOVE_DUR + STIM_PRESENT_DUR
        patch_start = SERVO_MOVE_DUR/1000.0
        patch_end = (SERVO_MOVE_DUR + STIM_PRESENT_DUR)/1000.0
        ax.axvspan(patch_start, patch_end, alpha=0.3, color='grey', zorder=-2)
        
        # Add reference lines
        ax.axvline(x=REWARD_TIME_OFFSET/1000.0, color='red', linestyle='--', alpha=0.7, label='Reward Time', zorder=-1)
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Lick Rate (Hz)')
        ax.set_title(f'Lick Rate Over Time - Day {results["day"]}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set y-axis limits to 0-12 Hz
        ax.set_ylim(0, 12)
        
        # Create subfolder for daily average lick rate plots (across all mice)
        daily_avg_folder = os.path.join(OUTPUT_FOLDER, 'daily_avg_lick_rates')
        os.makedirs(daily_avg_folder, exist_ok=True)
        plot_file = f'{daily_avg_folder}/lick_rate_day_{results["day"]}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()  # Display interactive plot window
        print(f"Day {results['day']} plot saved to: {plot_file}")
        
        plt.close()  # Close figure to free memory

def plot_lick_rate_results_by_mouse(results_by_day):
    """
    Create lick rate over time plots organized by mouse (one plot per mouse, SSH compatible).
    Each mouse gets their own figure with subplots for each day.
    """
    print("=" * 60)
    print("CREATING LICK RATE PLOTS BY MOUSE")
    print("=" * 60)
    
    if not results_by_day:
        print("No lick rate data available for plotting.")
        return
    
    # Get all unique mouse names across all days
    all_mouse_names = set()
    for day_data in results_by_day.values():
        all_mouse_names.update(day_data['cs_plus_rates_by_mouse'].keys())
        all_mouse_names.update(day_data['cs_minus_rates_by_mouse'].keys())
    
    mouse_names = sorted(list(all_mouse_names))
    
    # Reorganize data by mouse
    data_by_mouse = {}
    for mouse_name in mouse_names:
        data_by_mouse[mouse_name] = {}
    
    # Extract data for each mouse from each day
    for day_idx, day_data in results_by_day.items():
        time_centers = day_data['time_centers'] / 1000.0  # Convert to seconds
        cs_plus_rates_by_mouse = day_data['cs_plus_rates_by_mouse']
        cs_minus_rates_by_mouse = day_data['cs_minus_rates_by_mouse']
        day_num = day_data['day']
        
        # Store data for each mouse that has data on this day
        for mouse_name in mouse_names:
            if mouse_name in cs_plus_rates_by_mouse:
                cs_plus_data = cs_plus_rates_by_mouse[mouse_name]
                cs_minus_data = cs_minus_rates_by_mouse.get(mouse_name, {'per_trial': np.array([]), 'avg': np.array([])})
                
                data_by_mouse[mouse_name][day_num] = {
                    'time_centers': time_centers,
                    'cs_plus_rates': cs_plus_data['avg'],
                    'cs_plus_rates_per_trial': cs_plus_data['per_trial'],
                    'cs_minus_rates': cs_minus_data['avg'],
                    'cs_minus_rates_per_trial': cs_minus_data['per_trial']
                }
    
    # Create plots for each mouse
    for mouse_name in mouse_names:
        mouse_data = data_by_mouse[mouse_name]
        if not mouse_data:  # Skip if no data for this mouse
            print(f"No data available for {mouse_name}, skipping...")
            continue
            
        print(f"Creating plot for {mouse_name}...")
        
        # Get available days for this mouse
        available_days = sorted(mouse_data.keys())
        n_days = len(available_days)
        
        if n_days == 0:
            continue
            
        # Calculate subplot grid (prefer more columns than rows)
        if n_days <= 3:
            n_cols = n_days
            n_rows = 1
        elif n_days <= 6:
            n_cols = 3
            n_rows = 2
        elif n_days <= 9:
            n_cols = 3
            n_rows = 3
        else:
            n_cols = 4
            n_rows = int(np.ceil(n_days / n_cols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle(f'Lick Rate Over Time - {mouse_name}', fontsize=16, y=0.98)
        
        # Handle single subplot case
        if n_days == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_days == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each day
        for plot_idx, day_num in enumerate(available_days):
            ax = axes[plot_idx]
            day_data = mouse_data[day_num]
            
            time_centers = day_data['time_centers']
            cs_plus_rates = day_data['cs_plus_rates']
            cs_plus_rates_per_trial = day_data['cs_plus_rates_per_trial']
            cs_minus_rates = day_data['cs_minus_rates']
            cs_minus_rates_per_trial = day_data['cs_minus_rates_per_trial']
            
            # Plot CS+ data with SEM
            if cs_plus_rates_per_trial.size > 0:
                # Calculate SEM across trials
                cs_plus_sem = np.std(cs_plus_rates_per_trial, axis=0) / np.sqrt(cs_plus_rates_per_trial.shape[0])
                
                # Plot average with shaded SEM
                ax.plot(time_centers, cs_plus_rates, 'b-', linewidth=2, label='CS+')
                ax.fill_between(time_centers, cs_plus_rates - cs_plus_sem, cs_plus_rates + cs_plus_sem, 
                                color='blue', alpha=0.2)
            
            # Plot CS- data with SEM if available
            if cs_minus_rates_per_trial.size > 0:
                # Calculate SEM across trials
                cs_minus_sem = np.std(cs_minus_rates_per_trial, axis=0) / np.sqrt(cs_minus_rates_per_trial.shape[0])
                
                # Plot average with shaded SEM
                ax.plot(time_centers, cs_minus_rates, 'k-', linewidth=2, label='CS-')
                ax.fill_between(time_centers, cs_minus_rates - cs_minus_sem, cs_minus_rates + cs_minus_sem, 
                                color='black', alpha=0.2)
            
            # Add simple grey patch from SERVO_MOVE_DUR to SERVO_MOVE_DUR + STIM_PRESENT_DUR
            patch_start = SERVO_MOVE_DUR/1000.0
            patch_end = (SERVO_MOVE_DUR + STIM_PRESENT_DUR)/1000.0
            ax.axvspan(patch_start, patch_end, alpha=0.3, color='grey', zorder=-2)
            
            # Add reference lines
            ax.axvline(x=REWARD_TIME_OFFSET/1000.0, color='white', alpha=1, zorder=-1) 
            #ax.axvline(x=REWARD_TIME_OFFSET/1000.0, color='turquoise', linestyle='--', alpha=1, label='Reward Time')

            # Formatting
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Lick Rate (Hz)')
            ax.set_title(f'Day {day_num}')
            #if day_num == 6: THIS IS IF YOU INCLUDE CL044 GO NO GO SESSION
             #   ax.set_title('Day 5 - go/no-go')
            ax.grid(False)
            ax.set_ylim(0, 12)
            
            # Add legend only to first subplot
            if plot_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for plot_idx in range(len(available_days), len(axes)):
            axes[plot_idx].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create subfolder for individual mouse subplot lick rate plots
        mouse_subplot_folder = os.path.join(OUTPUT_FOLDER, 'mouse_subplot_lickrate_all_days')
        os.makedirs(mouse_subplot_folder, exist_ok=True)
        plot_file = f'{mouse_subplot_folder}/lick_rate_{mouse_name}_all_days.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()  # Display interactive plot window
        print(f"{mouse_name} plot saved to: {plot_file}")
        
        plt.close()  # Close figure to free memory

#%% MAIN EXECUTION

def show_analysis_menu():
    """
    Display an interactive menu for choosing which analyses to run.
    """
    print("\n" + "="*60)
    print("PAVLOVIAN TEXTURE ANALYSIS - INTERACTIVE MENU")
    print("="*60)
    print("Available Analyses:")
    print("1. Lick Index Analysis")
    print("2. Anticipatory Licking Analysis") 
    print("3. Average Lick Rate Over Time Analysis")
    print("4. Lick Rate by Mouse Analysis")
    print("5. Run ALL analyses")
    print("6. Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("Invalid choice. Please enter 1-6.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return '6'

def main():
    """
    Main function to run the analysis based on user choice.
    """
    # Show configuration
    print("PAVLOVIAN TEXTURE ANALYSIS")
    print("=" * 60)
    print(f"Data Directory: {DATA_DIRECTORY}")
    print(f"Analyzing Mice: {MOUSE_FOLDERS}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print("=" * 60)
    
    # Get user choice
    choice = show_analysis_menu()
    
    if choice == '6':
        print("Exiting analysis.")
        return
    
    # Ensure directories exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load data (now returns whisker trim filepaths separately)
    print("\nLoading data files...")
    filepaths_df, whiskertrim_filepaths, mouse_names = get_mouse_filepaths(DATA_DIRECTORY, MOUSE_FOLDERS)
    
    if filepaths_df.empty:
        print("ERROR: No data files found. Exiting.")
        return

    # Run selected analyses
    if choice in ['1', '5']:  # Lick Index Analysis
        print("\n" + "="*40)
        print("RUNNING LICK INDEX ANALYSIS")
        print("="*40)
        lick_index_results = run_lick_index_analysis(filepaths_df, whiskertrim_filepaths, mouse_names)
        plot_lick_index_results(lick_index_results, filepaths_df.shape[0])

    if choice in ['2', '5']:  # Anticipatory Licking Analysis
        print("\n" + "="*40)
        print("RUNNING ANTICIPATORY LICKING ANALYSIS")
        print("="*40)
        anticipatory_licking_results = run_anticipatory_licking_analysis(filepaths_df, whiskertrim_filepaths, mouse_names)
        plot_anticipatory_licking_results(anticipatory_licking_results, filepaths_df.shape[0])

    if choice in ['3', '5']:  # Average Lick Rate Analysis
        print("\n" + "="*40)
        print("RUNNING AVERAGE LICK RATE ANALYSIS")
        print("="*40)
        lick_rate_results = run_lick_rate_analysis(filepaths_df, mouse_names)
        plot_lick_rate_avg(lick_rate_results)

    if choice in ['4', '5']:  # Lick Rate by Mouse Analysis
        print("\n" + "="*40)
        print("RUNNING LICK RATE BY MOUSE ANALYSIS")
        print("="*40)
        if choice == '4' or not (choice == '5' and choice in ['3', '5']):
            # Need to run the analysis if it wasn't already run
            lick_rate_results = run_lick_rate_analysis(filepaths_df, mouse_names)
        plot_lick_rate_results_by_mouse(lick_rate_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print("="*60)

if __name__ == "__main__":
    main()

