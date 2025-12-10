"""
Individual Trial Heatmap Generator

This script creates heatmaps showing individual trial dF/F responses for all ROIs.
- X-axis: Time (matching plot_individual_trial_traces limits)
- Y-axis: ROI number
- Each row shows one ROI's response for that specific trial
- lick times are plotted as vertical lines above each heatmap
- Separate figures: CS+ trials and CS- trials
- Easy to configure which trials to plot (default: first 5 of each type)
- Saves to /heatmaps/individual_trial_heatmaps/animalname/
- data path and save path are set in set_paths.py

Usage:
    python individual_trial_heatmaps.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm

# Import shared data loading functions
from data_loader import load_processed_data
from extract_trial_dff import extract_trial_dff

# Add the data_processing directory to Python path for the import
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)

from learning_phase_mapping import convert_animal_data_to_phases  # noqa: E402
from set_paths import processed_data_path, save_figures_path  # noqa: E402


# Which trials do you want to plot? (0-indexed, exclusive end)
CS_PLUS_TRIAL_RANGE = (0, 6)   # trials 1-5 CS+ trials 
CS_MINUS_TRIAL_RANGE = (0, 6)  # trials 1-5 CS- trials


def find_peak_timing_for_rois(trial_averaged_responses, time_axis):
    """
    Find the peak timing for each ROI based on the maximum amplitude during the plot duration.
    Uses a rolling average of 3 frames to smooth the signal before finding peaks.
    
    Args:
        trial_averaged_responses (np.ndarray): Array of shape (n_rois, n_frames) - averaged responses
        time_axis (np.ndarray): Time axis in seconds
        
    Returns:
        np.ndarray: Array of peak time indices for each ROI (earlier peaks have smaller indices)
    """
    n_rois, n_frames = trial_averaged_responses.shape
    peak_timings = np.zeros(n_rois, dtype=int)
    
    for roi_idx in range(n_rois):
        roi_trace = trial_averaged_responses[roi_idx, :]
        
        # Apply rolling average of 3 frames
        # Use numpy convolution for rolling average
        window_size = 3
        kernel = np.ones(window_size) / window_size
        
        # Pad the trace to handle edges (mode='same' keeps original length)
        smoothed_trace = np.convolve(roi_trace, kernel, mode='same')
        
        # Find the frame with maximum absolute amplitude in the smoothed trace
        # This handles both positive and negative peaks
        peak_frame = np.argmax(np.abs(smoothed_trace))
        peak_timings[roi_idx] = peak_frame
    
    return peak_timings


def sort_rois_by_peak_timing(trial_averaged_responses, time_axis):
    """
    Sort ROIs by their peak timing. Later peaks get higher y positions (lower indices).
    
    Args:
        trial_averaged_responses (np.ndarray): Array of shape (n_rois, n_frames) - averaged responses
        time_axis (np.ndarray): Time axis in seconds
        
    Returns:
        tuple: (sorted_responses, roi_order)
            - sorted_responses: Array of shape (n_rois, n_frames) with ROIs reordered
            - roi_order: Array of original ROI indices in the new order
    """
    peak_timings = find_peak_timing_for_rois(trial_averaged_responses, time_axis)
    
    # Sort by peak timing (later peaks first)
    # np.argsort gives indices that would sort the array in ascending order
    # We reverse the order to get later peaks first (descending order)
    roi_order = np.argsort(peak_timings)[::-1]
    
    # Reorder the responses according to the new ROI order
    sorted_responses = trial_averaged_responses[roi_order, :]
    
    return sorted_responses, roi_order


def extract_trial_lick_frames(lick_frames, stim_frames, pre_frames=30, post_frames=180):
    """
    Extract lick frames for each trial around stimulus presentation.
    
    Args:
        lick_frames (np.ndarray): Array of frame numbers when licks occurred
        stim_frames (list): List of (start_frame, stop_frame) tuples for each trial
        pre_frames (int): Number of frames before stimulus onset
        post_frames (int): Number of frames after stimulus onset
        
    Returns:
        list: List of arrays, each containing lick frame indices relative to trial start for each trial
    """
    trial_lick_frames = []
    
    for start_frame, stop_frame in stim_frames:
        # Define trial window around stimulus onset
        trial_start = max(0, start_frame - pre_frames)
        trial_end = min(start_frame + post_frames, start_frame + post_frames)  # Use post_frames from stimulus onset
        
        # Find licks within this trial window
        trial_licks = lick_frames[(lick_frames >= trial_start) & (lick_frames <= trial_end)]
        
        # Convert to relative frame indices within the trial window
        relative_lick_frames = trial_licks - trial_start
        
        trial_lick_frames.append(relative_lick_frames)
    
    return trial_lick_frames


def create_individual_trial_heatmaps(animal_data, animal_name, session_idx, save_path=None, figsize=(12, 8)):
    """
    Create individual trial heatmaps for specified trials in a single session.
    
    Args:
        animal_data (dict): Animal data dictionary
        animal_name (str): Name of the animal
        session_idx (int): Session index to analyze
        save_path (str, optional): Path to save the heatmaps
        figsize (tuple): Figure size for each heatmap
    """
    if session_idx not in animal_data['sessions']:
        print(f"Error: Session {session_idx} not found in data for {animal_name}")
        available_sessions = list(animal_data['sessions'].keys())
        print(f"Available sessions: {available_sessions}")
        return
    
    session_data = animal_data['sessions'][session_idx]
    dff = session_data['dff']
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    lick_frames = session_data.get('lick_frames', np.array([]))  # Default to empty array if no licks
    
    print(f"Processing {animal_name}, Session {session_idx}")
    print(f"  {dff.shape[0]} ROIs, {len(cs_plus_frames)} CS+ trials, {len(cs_minus_frames)} CS- trials")
    print(f"  {len(lick_frames)} lick events")
    
    # Extract trial responses
    cs_plus_trials, time_axis = extract_trial_dff(dff, cs_plus_frames)
    cs_minus_trials, _ = extract_trial_dff(dff, cs_minus_frames)
    
    # Extract trial lick frames
    cs_plus_licks = extract_trial_lick_frames(lick_frames, cs_plus_frames)
    cs_minus_licks = extract_trial_lick_frames(lick_frames, cs_minus_frames)
    
    # Determine which trials to plot
    cs_plus_start, cs_plus_end = CS_PLUS_TRIAL_RANGE
    cs_minus_start, cs_minus_end = CS_MINUS_TRIAL_RANGE
    
    # Ensure we don't exceed available trials
    cs_plus_end = min(cs_plus_end, len(cs_plus_frames))
    cs_minus_end = min(cs_minus_end, len(cs_minus_frames))
    
    if cs_plus_start >= len(cs_plus_frames):
        print(f"Warning: CS+ trial range starts at {cs_plus_start} but only {len(cs_plus_frames)} CS+ trials available")
        return
    if cs_minus_start >= len(cs_minus_frames):
        print(f"Warning: CS- trial range starts at {cs_minus_start} but only {len(cs_minus_frames)} CS- trials available")
        return
    
    cs_plus_trials_to_plot = range(cs_plus_start, cs_plus_end)
    cs_minus_trials_to_plot = range(cs_minus_start, cs_minus_end)
    
    print(f"  Plotting CS+ trials: {list(cs_plus_trials_to_plot)} (trials {cs_plus_start+1}-{cs_plus_end})")
    print(f"  Plotting CS- trials: {list(cs_minus_trials_to_plot)} (trials {cs_minus_start+1}-{cs_minus_end})")
    print(f"  ROIs will be sorted individually for each trial based on peak timing")
    
    # Create heatmaps (ROI sorting will be done individually for each trial)
    create_heatmap_figure(cs_plus_trials, None, time_axis, cs_plus_trials_to_plot, 
                         animal_name, session_idx, "CS+", save_path, figsize, cs_plus_licks)
    
    create_heatmap_figure(cs_minus_trials, None, time_axis, cs_minus_trials_to_plot, 
                         animal_name, session_idx, "CS-", save_path, figsize, cs_minus_licks)


def create_heatmap_figure(trial_data, roi_order, time_axis, trials_to_plot, 
                         animal_name, session_idx, stimulus_type, save_path, figsize, trial_licks):
    """
    Create a single heatmap figure with individual trials as subplots.
    Each trial is sorted individually based on peak timing in that specific trial.
    Includes lick events plotted as vertical lines above each heatmap.
    
    Args:
        trial_data (np.ndarray): Trial data of shape (n_trials, n_rois, n_frames)
        roi_order (np.ndarray): ROI ordering for sorting (ignored - each trial sorted individually)
        time_axis (np.ndarray): Time axis in seconds
        trials_to_plot (range): Range of trial indices to plot
        animal_name (str): Name of the animal
        session_idx (int): Session index
        stimulus_type (str): "CS+" or "CS-"
        save_path (str, optional): Path to save the figure, comes from set_paths.py
        figsize (tuple): Figure size
        trial_licks (list): List of arrays containing lick frame indices for each trial
    """
    n_trials = len(trials_to_plot)
    if n_trials == 0:
        print(f"No {stimulus_type} trials to plot")
        return
    
    # Calculate subplot layout (prefer wider layouts)
    if n_trials <= 3:
        n_cols = n_trials
        n_rows = 1
    elif n_trials <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_trials + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with GridSpec for better control over subplot positioning
    # Each trial will have 2 subplots: one for licks (top, small) and one for heatmap (bottom, larger)
    from matplotlib.gridspec import GridSpec
    
    # Make plots smaller and add more spacing
    fig = plt.figure(figsize=(figsize[0] * 0.8, figsize[1] * n_rows / 2.2))
    
    # Create GridSpec with 2 rows per trial row (lick subplot + heatmap subplot)
    # Add extra spacing between trial rows by including spacer rows
    if n_rows == 1:
        # Single row case
        gs = GridSpec(2, n_cols, figure=fig, 
                      height_ratios=[0.08, 1],  # Smaller lick subplot, larger heatmap
                      hspace=0.08,  # Small spacing between lick subplot and heatmap
                      wspace=0.3,  # Horizontal spacing between columns
                      top=0.88,    # More space below title
                      bottom=0.1,  # Space for labels
                      left=0.08,   # Space for y-labels
                      right=0.85)  # Space for colorbar
    else:
        # Multiple rows case - add spacer rows between trial rows
        total_grid_rows = n_rows * 2 + (n_rows - 1)  # 2 rows per trial + spacers between trials
        height_ratios = []
        
        for i in range(n_rows):
            height_ratios.extend([0.08, 1])  # Lick subplot (smaller), heatmap
            if i < n_rows - 1:  # Add spacer between trial rows (except after last row)
                height_ratios.append(0.3)  # Spacer row
        
        gs = GridSpec(total_grid_rows, n_cols, figure=fig, 
                      height_ratios=height_ratios,
                      hspace=0.05,  # Minimal spacing within GridSpec
                      wspace=0.3,   # Horizontal spacing between columns
                      top=0.88,     # More space below title
                      bottom=0.1,   # Space for labels
                      left=0.08,    # Space for y-labels
                      right=0.85)   # Space for colorbar
    
    # Add overall title with more space below it
    fig.suptitle(f'{animal_name} - Session {session_idx} - Individual {stimulus_type} Trial Heatmaps', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Set up colormap normalization (fixed range)
    vmin_fixed = -0.05
    vmax_fixed = 0.5
    vcenter_fixed = vmin_fixed + (vmax_fixed-vmin_fixed)/2
    norm = TwoSlopeNorm(vmin=vmin_fixed, vcenter=vcenter_fixed, vmax=vmax_fixed)
    
    # Plot each trial
    for plot_idx, trial_idx in enumerate(trials_to_plot):
        # Calculate grid positions
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        
        # Calculate actual grid row indices accounting for spacer rows
        if n_rows == 1:
            lick_grid_row = 0
            heatmap_grid_row = 1
        else:
            # For multiple rows: each trial row takes 2 positions + 1 spacer (except last)
            # Row 0: grid positions 0 (lick), 1 (heatmap)
            # Row 1: grid positions 3 (lick), 4 (heatmap) [position 2 is spacer]
            # Row 2: grid positions 6 (lick), 7 (heatmap) [position 5 is spacer]
            lick_grid_row = row * 3  # 0, 3, 6, ...
            heatmap_grid_row = row * 3 + 1  # 1, 4, 7, ...
        
        # Create lick subplot (top, small)
        lick_ax = fig.add_subplot(gs[lick_grid_row, col])
        
        # Create heatmap subplot (bottom, larger)
        heatmap_ax = fig.add_subplot(gs[heatmap_grid_row, col])
        
        # Get trial data and sort ROIs individually for this trial
        trial_response = trial_data[trial_idx]  # Shape: (n_rois, n_frames)
        
        # Sort ROIs by peak timing for this specific trial
        _, trial_roi_order = sort_rois_by_peak_timing(trial_response, time_axis)
        sorted_trial_response = trial_response[trial_roi_order, :]
        
        # === LICK SUBPLOT ===
        # Set up lick subplot with invisible axis
        lick_ax.set_xlim(0, len(time_axis))
        lick_ax.set_ylim(0, 1)
        
        # Hide all spines, ticks, and labels
        for spine in lick_ax.spines.values():
            spine.set_visible(False)
        lick_ax.set_xticks([])
        lick_ax.set_yticks([])
        lick_ax.set_xlabel('')
        lick_ax.set_ylabel('')
        
        # Add stimulus timing bar (above lick lines)
        onset_idx = np.argmin(np.abs(time_axis))  # t=0
        offset_idx = np.argmin(np.abs(time_axis - 2.0))  # t=2s (typical stimulus duration)
        lick_ax.plot([onset_idx, offset_idx], [0.95, 0.95], # this controls the vertical position of the stimulus bar
                    color='black', linewidth=3, solid_capstyle='butt')
        
        # Plot lick events as shorter vertical lines (below stimulus bar)
        if trial_idx < len(trial_licks) and len(trial_licks[trial_idx]) > 0:
            lick_frames_for_trial = trial_licks[trial_idx]
            # Filter licks to the same time window as the heatmap
            time_start_idx = np.argmin(np.abs(time_axis + 1))  # -1.0 seconds
            time_end_idx = np.argmin(np.abs(time_axis - 4.0))    # +3.0 seconds
            
            # Only show licks within the display window
            valid_licks = lick_frames_for_trial[(lick_frames_for_trial >= time_start_idx) & 
                                               (lick_frames_for_trial <= time_end_idx)]
            
            for lick_frame in valid_licks:
                # Draw shorter lines from bottom (0.1) to middle (0.6) of the subplot
                lick_ax.plot([lick_frame, lick_frame], [0.1, 0.6], 
                           color='black', linewidth=1.5, alpha=0.8)
        
        # Set same x-limits as heatmap
        time_start_idx = np.argmin(np.abs(time_axis + 1))  # -1.0 seconds
        time_end_idx = np.argmin(np.abs(time_axis - 4.0))    # +3.0 seconds
        lick_ax.set_xlim(time_start_idx, time_end_idx)
        
        # === HEATMAP SUBPLOT ===
        # Create heatmap
        im = heatmap_ax.imshow(sorted_trial_response, aspect='auto', cmap='jet', 
                              norm=norm, interpolation='nearest')
        
        # Set up axes
        # Time axis
        time_ticks = np.arange(0, len(time_axis), 30)  # Every 30 frames (1 second)
        time_labels = [f'{time_axis[i]:.0f}' for i in time_ticks]
        heatmap_ax.set_xticks(time_ticks)
        heatmap_ax.set_xticklabels(time_labels)
        heatmap_ax.set_xlabel('Time (s)')
        
        # ROI axis
        n_rois = len(trial_roi_order)
        roi_ticks = np.arange(0, n_rois, max(1, n_rois // 10))  # Show ~10 ticks
        heatmap_ax.set_yticks(roi_ticks)
        # Show original ROI numbers (1-based) in the new sorted order for this trial
        heatmap_ax.set_yticklabels([f'{trial_roi_order[i]+1}' for i in roi_ticks])
        heatmap_ax.set_ylabel('ROI #')
        
        # Set limits to match plot_individual_trial_traces
        heatmap_ax.set_xlim(time_start_idx, time_end_idx)
        heatmap_ax.set_ylim(-0.5, n_rois - 0.5)  # Center ROIs on pixel centers
    
    # Add colorbar (smaller, spanning just one subplot row)
    # Position colorbar to the right of the figure, smaller height
    if n_rows == 1:
        # Single row: center the colorbar
        cbar_ax = fig.add_axes([0.87, 0.35, 0.02, 0.3])  # [left, bottom, width, height]
    else:
        # Multiple rows: align with first row of heatmaps
        cbar_ax = fig.add_axes([0.87, 0.65, 0.02, 0.2])  # [left, bottom, width, height]
    
    cbar = fig.colorbar(im, cax=cbar_ax, norm=norm)
    cbar.set_label('dF/F', rotation=270, labelpad=15, fontsize=9)
    
    
    # Save the figure
    if save_path:
        # Create animal-specific folder within the save path
        animal_save_path = os.path.join(save_path, animal_name)
        os.makedirs(animal_save_path, exist_ok=True)
        filename = f'individual_trials_{animal_name}_session{session_idx}_{stimulus_type.replace("+", "plus").replace("-", "minus")}.png'
        filepath = os.path.join(animal_save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {stimulus_type} heatmap to {filepath}")
    
    plt.show()


def analyze_all_animals(data_path, save_path=None):
    """
    Create individual trial heatmaps for all animals and all their sessions.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str, optional): Path to save heatmaps
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    total_processed = 0
    total_sessions = 0
    
    # Process each animal
    for animal_name, animal_data in all_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {animal_name}")
        print(f"{'='*60}")
        
        available_sessions = sorted(list(animal_data['sessions'].keys()))
        print(f"Available sessions: {available_sessions}")
        
        animal_processed = 0
        
        # Process each session for this animal
        for session_idx in available_sessions:
            try:
                print(f"\n  Processing Session {session_idx}...")
                create_individual_trial_heatmaps(animal_data, animal_name, session_idx, save_path)
                animal_processed += 1
                total_processed += 1
            except Exception as e:
                print(f"  Error processing {animal_name} Session {session_idx}: {e}")
                continue
        
        total_sessions += len(available_sessions)
        print(f"\nCompleted {animal_processed}/{len(available_sessions)} sessions for {animal_name}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {total_processed}/{total_sessions} sessions")
    print(f"{'='*60}")


def analyze_single_animal_session(data_path, save_path=None):
    """
    Get user input for animal and session selection, then create heatmaps.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str, optional): Path to save heatmaps
        
    Returns:
        bool: True if successful, False if cancelled or failed
    """
    try:
        all_data = load_processed_data(data_path)
        available_animals = list(all_data.keys())
        
        print(f"\nAvailable animals: {available_animals}")
        
        # Get animal selection
        print("\nAvailable animals:")
        for i, animal in enumerate(available_animals, 1):
            print(f"{i}. {animal}")
        
        while True:
            try:
                animal_choice = input(f"\nSelect animal (1-{len(available_animals)}): ").strip()
                animal_idx = int(animal_choice) - 1
                if 0 <= animal_idx < len(available_animals):
                    selected_animal = available_animals[animal_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_animals)}")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid number or Ctrl+C to exit")
                return False
        
        # Get session selection
        animal_data = all_data[selected_animal]
        available_sessions = sorted(list(animal_data['sessions'].keys()))
        
        print(f"\nAvailable sessions for {selected_animal}: {available_sessions}")
        
        while True:
            try:
                session_input = input(f"\nEnter session number: ").strip()
                session_idx = int(session_input)
                if session_idx in available_sessions:
                    break
                else:
                    print(f"Please enter one of the available sessions: {available_sessions}")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid session number or Ctrl+C to exit")
                return False
        
        # Process the selected animal and session
        print(f"\n{'='*60}")
        print(f"Processing {selected_animal}, Session {session_idx}")
        print(f"{'='*60}")
        
        create_individual_trial_heatmaps(animal_data, selected_animal, session_idx, save_path)
        
        print(f"\n{'='*60}")
        print("Completed processing!")
        print(f"{'='*60}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please check the path and try again.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Get paths from centralized config
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'heatmaps','individual_trial_heatmaps')
    
    print("=" * 60)
    print("INDIVIDUAL TRIAL HEATMAP GENERATOR")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  CS+ trials to plot: {CS_PLUS_TRIAL_RANGE[0]+1}-{CS_PLUS_TRIAL_RANGE[1]} (trials {CS_PLUS_TRIAL_RANGE[0]+1} to {CS_PLUS_TRIAL_RANGE[1]})")
    print(f"  CS- trials to plot: {CS_MINUS_TRIAL_RANGE[0]+1}-{CS_MINUS_TRIAL_RANGE[1]} (trials {CS_MINUS_TRIAL_RANGE[0]+1} to {CS_MINUS_TRIAL_RANGE[1]})")
    print(f"  (Edit CS_PLUS_TRIAL_RANGE and CS_MINUS_TRIAL_RANGE at the top of the script to change)")
    
    try:
        # Load data to show available animals
        all_data = load_processed_data(data_path)
        available_animals = list(all_data.keys())
        
        print(f"\nAvailable animals: {available_animals}")
        
        print("\nChoose analysis mode:")
        print("1. Single animal, single session")
        print("2. All animals, all sessions")
        
        while True:
            try:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice in ['1', '2']:
                    break
                else:
                    print("Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nExiting...")
                exit()
        
        if choice == '1':
            # Single animal, single session analysis
            success = analyze_single_animal_session(data_path, save_path)
            if not success:
                print("Analysis cancelled or failed.")
                
        else:
            # All animals, all sessions analysis
            print(f"\nThis will process ALL animals and ALL their sessions.")
            print(f"This may take a long time and generate many files.")
            print("Continue? (y/n): ", end="")
            confirm = input().strip().lower()
            if confirm in ['y', 'yes']:
                analyze_all_animals(data_path, save_path)
            else:
                print("Analysis cancelled.")
                
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"Error: {e}")
