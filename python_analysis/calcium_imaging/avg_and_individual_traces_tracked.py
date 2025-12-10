"""
Only plots tracked ROIs. 

Average a single ROI's df/f traces across all CS+ and CS- trials within a session.
Makes a figure for each ROI, with a subplot for CS+ and CS- across all sessions 

Calculates selectivity index for each ROI in each session and includes it in the figure

User can input if they want all animals for all sessions or a single animal for a single session
This does not include individual trial traces - you can find that in plot_avg_trace_with_inidivdual_traces.py

Data path and save path are set in set_paths.py

"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from datetime import datetime
# Add the data_processing directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)

from learning_phase_mapping import convert_animal_data_to_phases  # noqa: E402
from set_paths import processed_data_path, save_figures_path  # noqa: E402
from extract_trial_dff import extract_trial_dff, extract_trial_dff_baseline_corrected  # noqa: E402
from data_loader import load_processed_data  # noqa: E402
from calculate_selectivity import calculate_roi_averaged_traces, calculate_selectivity_index  # noqa: E402

def calculate_selectivity_index_for_roi(roi_dff, cs_plus_frames, cs_minus_frames, 
                                       response_window_start=0, response_window_end=2, 
                                       frame_rate=30, method='peak'):
    """
    Calculate selectivity index for a single ROI using baseline-corrected data.
    Refactored to use centralized functions from calculate_selectivity.py
    
    Args:
        roi_dff (np.ndarray): dF/F data for single ROI (1 x time)
        cs_plus_frames (list): List of (start_frame, stop_frame) tuples for CS+ trials
        cs_minus_frames (list): List of (start_frame, stop_frame) tuples for CS- trials
        response_window_start (float): Start of response window in seconds (relative to stimulus onset)
        response_window_end (float): End of response window in seconds (relative to stimulus onset)
        frame_rate (int): Frame rate in Hz
        method (str): Method for calculating response values ('peak' or 'mean')
        
    Returns:
        float: Selectivity index for the ROI
    """
    # Extract baseline-corrected trial responses
    cs_plus_trials, time_axis = extract_trial_dff_baseline_corrected(roi_dff, cs_plus_frames)
    cs_minus_trials, _ = extract_trial_dff_baseline_corrected(roi_dff, cs_minus_frames)
    
    # Calculate ROI-averaged traces using centralized function
    cs_plus_data = calculate_roi_averaged_traces(cs_plus_trials, roi_indices=[0])
    cs_minus_data = calculate_roi_averaged_traces(cs_minus_trials, roi_indices=[0])
    
    # Calculate selectivity index using centralized function
    selectivity_result = calculate_selectivity_index(
        cs_plus_data, 
        cs_minus_data,
        response_window_start=response_window_start,
        response_window_end=response_window_end,
        cs_plus_frames=cs_plus_frames,
        cs_minus_frames=cs_minus_frames,
        frame_rate=frame_rate,
        method=method
    )
    
    # Extract the selectivity value for ROI index 0
    return selectivity_result[0]['selectivity']

def get_tracked_roi_data_for_session(animal_data, session_idx, tracked_roi_idx):
    """
    Get data for a specific tracked ROI from a specific session.
    
    Args:
        animal_data (dict): Animal data containing sessions and tracked ROIs
        session_idx (int): Session index
        tracked_roi_idx (int): Index of the tracked ROI (0-based)
        
    Returns:
        dict: Session data with ROI-specific information, or None if data not available
    """
    # Check if session exists
    if session_idx not in animal_data['sessions']:
        return None
    
    session_data = animal_data['sessions'][session_idx]
    
    # Check if tracked ROIs exist
    if 'Tracked ROIs' not in animal_data:
        print(f"No tracked ROIs found for this animal")
        return None
    
    tracked_indices = animal_data['Tracked ROIs']
    
    # Check if tracked ROI index is valid
    if tracked_roi_idx >= tracked_indices.shape[0]:
        return None
    
    # Get the actual ROI index for this session (convert from 1-based to 0-based)
    actual_roi_idx = tracked_indices[tracked_roi_idx, session_idx]
    
    # Check if this ROI exists in this session (NaN means no match)
    if np.isnan(actual_roi_idx):
        return None
    
    # Convert to 0-based indexing
    actual_roi_idx = int(actual_roi_idx) - 1
    
    # Check if ROI index is valid for this session's data
    if actual_roi_idx >= session_data['dff'].shape[0] or actual_roi_idx < 0:
        return None
    
    # Extract data for this specific ROI
    roi_dff = session_data['dff'][actual_roi_idx:actual_roi_idx+1, :]  # Keep as 2D array
    
    return {
        'dff': roi_dff,
        'cs_plus_frames': session_data['cs_plus_frames'],
        'cs_minus_frames': session_data['cs_minus_frames'],
        'label': session_data['label'],
        'actual_roi_idx': actual_roi_idx
    }

def plot_tracked_roi_longitudinal_traces(animal_data, tracked_roi_idx, save_path=None, 
                                       figsize=(10, 20), animal_name=None):
    """
    Plot longitudinal traces for a single tracked ROI across all sessions.
    Creates a dynamic subplot layout based on available sessions: Columns are CS+ and CS-, Rows are sessions.
    
    Args:
        animal_data (dict): Animal data containing sessions and tracked ROIs
        tracked_roi_idx (int): Index of the tracked ROI to plot (0-based)
        save_path (str, optional): Path to save plots
        figsize (tuple): Figure size
        animal_name (str, optional): Name of the animal for the title
    """
    # Convert day labels to learning phases and get ordered labels
    session_mapping, session_labels = convert_animal_data_to_phases(animal_data)
    n_sessions = len(session_labels)
    
    # Create subplots with dynamic number of rows based on available sessions
    fig, axes = plt.subplots(n_sessions, 2, figsize=(figsize[0], figsize[1] * n_sessions / 4))
    
    # Add overall title
    if animal_name:
        fig.suptitle(f'{animal_name} - Tracked ROI {tracked_roi_idx + 1}', fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Tracked ROI {tracked_roi_idx + 1}', fontsize=16, fontweight='bold')
    
    # Add column headers for CS+ and CS-
    axes[0, 0].text(0.5, 1.15, 'CS+', transform=axes[0, 0].transAxes, 
                   fontsize=16, fontweight='bold', 
                   horizontalalignment='center', verticalalignment='center')
    axes[0, 1].text(0.5, 1.15, 'CS-', transform=axes[0, 1].transAxes, 
                   fontsize=16, fontweight='bold', 
                   horizontalalignment='center', verticalalignment='center')
    
    for plot_idx, session_label in enumerate(session_labels):
        session_idx = session_mapping[session_label]
        # Get axes for CS+ (left column) and CS- (right column)
        ax_plus = axes[plot_idx, 0]  # Left column for CS+
        ax_minus = axes[plot_idx, 1]  # Right column for CS-
        
        # Get data for this session and ROI
        roi_session_data = get_tracked_roi_data_for_session(animal_data, session_idx, tracked_roi_idx)
        
        if roi_session_data is None:
            # No data available - leave subplots empty but labeled
            for ax in [ax_plus, ax_minus]:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('dF/F')
                ax.set_xlim(-1.0, 4)
                ax.set_ylim(-0.5, 3)
            
            # Add session label on the left
            ax_plus.text(-0.25, 0.5, f'{session_label}\n(No data)', transform=ax_plus.transAxes, 
                        fontsize=12, fontweight='bold', rotation=90, 
                        verticalalignment='center', horizontalalignment='center')
            continue
        
        # Extract trial responses for CS+ and CS-
        cs_plus_trials, time_axis = extract_trial_dff(roi_session_data['dff'], 
                                                           roi_session_data['cs_plus_frames'])
        cs_minus_trials, _ = extract_trial_dff(roi_session_data['dff'], 
                                                   roi_session_data['cs_minus_frames'])
        
        # Get data for the single ROI (index 0 since we extracted only one ROI)
        cs_plus_roi_trials = cs_plus_trials[:, 0, :]  # Shape: (n_trials, n_frames)
        cs_minus_roi_trials = cs_minus_trials[:, 0, :]  # Shape: (n_trials, n_frames)
        
        # Add stimulus time shaded patch (behind all traces) for both subplots
        for ax in [ax_plus, ax_minus]:
            ax.axvspan(xmin=0, xmax=2, color='pink', alpha=0.2, zorder=0)
        
        # Add reward line for CS+ trials (only on CS+ subplot)
        ax_plus.axvline(x=2, color='k', linestyle='--', alpha=0.7, linewidth=1)
        
        # Plot CS+ trials on left subplot
        ax_plus.plot(time_axis, cs_plus_roi_trials.T, 
                    color='steelblue', alpha=0.6, linewidth=0.5)
        
        # Plot CS- trials on right subplot
        ax_minus.plot(time_axis, cs_minus_roi_trials.T, 
                     color='grey', alpha=0.6, linewidth=0.5)
        
        # Calculate and plot average traces
        cs_plus_mean = np.mean(cs_plus_roi_trials, axis=0)
        cs_minus_mean = np.mean(cs_minus_roi_trials, axis=0)
        
        # Plot thick average traces
        ax_plus.plot(time_axis, cs_plus_mean, 'b-', linewidth=3, label='CS+ (avg)', alpha=0.9)
        ax_minus.plot(time_axis, cs_minus_mean, 'k-', linewidth=3, label='CS- (avg)', alpha=0.9)
        
        # Calculate selectivity index for this ROI in this session
        try:
            selectivity = calculate_selectivity_index_for_roi(
                roi_session_data['dff'], 
                roi_session_data['cs_plus_frames'], 
                roi_session_data['cs_minus_frames']
            )
            
            # Format selectivity index text
            if np.isnan(selectivity):
                si_text = 'SI: NaN'
            elif np.isinf(selectivity):
                if selectivity > 0:
                    si_text = 'SI: +Inf'
                else:
                    si_text = 'SI: -Inf'
            else:
                si_text = f'SI: {selectivity:.3f}'
                
            # Add selectivity index to upper right corner of CS+ subplot
            ax_plus.text(0.95, 0.95, si_text, 
                        transform=ax_plus.transAxes, fontsize=12, 
                        verticalalignment='top', horizontalalignment='right')
                        
        except Exception as e:
            print(f"    Warning: Could not calculate selectivity index for {session_label}: {e}")
        
        # Set axis properties for both subplots
        for ax in [ax_plus, ax_minus]:
            ax.set_ylim(-0.5, 3)
            ax.set_xlim(-1.0, 4)
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel('dF/F', fontsize=14)
            
            # Remove top and right spines (box), keep only x and y axis lines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        
        # Add session labels on the left side
        ax_plus.text(-0.25, 0.5, session_label, transform=ax_plus.transAxes, 
                    fontsize=12, fontweight='bold', rotation=90, 
                    verticalalignment='center', horizontalalignment='center')
        
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f'{animal_name}_tracked_ROI_{tracked_roi_idx + 1}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches='tight')
        print(f"Saved tracked ROI plot to {save_path}")
    
    plt.show()
    plt.close()

def analyze_tracked_rois_for_animal(data_path, animal_name, save_path=None):
    """
    Analyze and plot longitudinal traces for all tracked ROIs of a specific animal.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        save_path (str, optional): Path to save plots
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    # Check if animal exists
    if animal_name not in all_data:
        available_animals = list(all_data.keys())
        raise ValueError(f"Animal '{animal_name}' not found. Available animals: {available_animals}")
    
    animal_data = all_data[animal_name]
    
    # Check if tracked ROIs exist
    if 'Tracked ROIs' not in animal_data:
        raise ValueError(f"No tracked ROIs found for animal '{animal_name}'")
    
    tracked_indices = animal_data['Tracked ROIs']
    n_tracked_rois = tracked_indices.shape[0]
    
    print(f"Processing {n_tracked_rois} tracked ROIs for {animal_name}...")
    
    # Create save path for this animal
    if save_path:
        animal_save_path = os.path.join(save_path, "tracked_ROIs", animal_name)
        os.makedirs(animal_save_path, exist_ok=True)
    else:
        animal_save_path = None
    
    # Plot each tracked ROI
    for roi_idx in range(n_tracked_rois):
        print(f"  Processing tracked ROI {roi_idx + 1}/{n_tracked_rois}...")
        
        try:
            plot_tracked_roi_longitudinal_traces(
                animal_data=animal_data,
                tracked_roi_idx=roi_idx,
                save_path=animal_save_path,
                animal_name=animal_name
            )
        except Exception as e:
            print(f"    Error processing tracked ROI {roi_idx + 1}: {e}")
            continue
    
    print(f"Completed analysis for {animal_name}")

def analyze_all_animals_tracked_rois(data_path, save_path=None):
    """
    Analyze and plot longitudinal traces for tracked ROIs of all animals.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str, optional): Path to save plots
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Process each animal
    for animal_name, animal_data in all_data.items():
        print(f"\nProcessing {animal_name}...")
        
        # Check if tracked ROIs exist for this animal
        if 'Tracked ROIs' not in animal_data:
            print(f"  No tracked ROIs found for {animal_name}. Skipping.")
            continue
        
        try:
            analyze_tracked_rois_for_animal(data_path, animal_name, save_path)
        except Exception as e:
            print(f"  Error processing {animal_name}: {e}")
            continue

def list_available_animals_and_sessions(data_path):
    """
    List all available animals and their sessions in the dataset.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        
    Returns:
        dict: Dictionary with animal names as keys and lists of session labels as values
    """
    all_data = load_processed_data(data_path)
    
    animal_sessions = {}
    animals_with_tracked_rois = {}
    
    for animal_name, animal_data in all_data.items():
        sessions = animal_data['sessions']
        session_labels = [session_data['label'] for session_data in sessions.values()]
        animal_sessions[animal_name] = session_labels
        
        # Check if animal has tracked ROIs
        if 'Tracked ROIs' in animal_data:
            n_tracked_rois = animal_data['Tracked ROIs'].shape[0]
            animals_with_tracked_rois[animal_name] = n_tracked_rois
    
    return animal_sessions, animals_with_tracked_rois

def print_available_data(data_path):
    """
    Print all available animals and sessions with tracked ROIs.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
    """
    animal_sessions, animals_with_tracked_rois = list_available_animals_and_sessions(data_path)
    
    print("Available animals and sessions:")
    print("=" * 50)
    for animal_name, sessions in animal_sessions.items():
        if animal_name in animals_with_tracked_rois:
            n_rois = animals_with_tracked_rois[animal_name]
            print(f"{animal_name} ({n_rois} tracked ROIs):")
        else:
            print(f"{animal_name} (NO tracked ROIs):")
        
        for session in sessions:
            print(f"  - {session}")
        print()
    
    print(f"Animals with tracked ROIs: {len(animals_with_tracked_rois)}")
    print(f"Total animals: {len(animal_sessions)}")

if __name__ == "__main__":
    # Get paths from centralized config
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'individual_trial_overlay_traces')
    
    # List all available animals and sessions
    print("=" * 60)
    print("LISTING AVAILABLE DATA")
    print("=" * 60)
    print_available_data(data_path)
    
    # Ask user to choose analysis mode
    print("\n" + "=" * 60)
    print("TRACKED ROI ANALYSIS")
    print("=" * 60)
    print("Choose analysis mode:")
    print("1. Single animal analysis")
    print("2. All animals with tracked ROIs analysis")
    
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
    
    analysis_mode = "single" if choice == "1" else "all"
    print(f"Selected mode: {'Single animal analysis' if analysis_mode == 'single' else 'All animals analysis'}")
    
    if analysis_mode == "single":
        # Single animal analysis mode
        print("=" * 60)
        print("SINGLE ANIMAL TRACKED ROI ANALYSIS MODE")
        print("=" * 60)
        
        # Get available animals with tracked ROIs
        animal_sessions, animals_with_tracked_rois = list_available_animals_and_sessions(data_path)
        
        if not animals_with_tracked_rois:
            print("No animals with tracked ROIs found!")
            exit()
        
        # Let user select animal
        print("\nAvailable animals with tracked ROIs:")
        animal_list = list(animals_with_tracked_rois.keys())
        for i, animal in enumerate(animal_list, 1):
            n_rois = animals_with_tracked_rois[animal]
            print(f"{i}. {animal} ({n_rois} tracked ROIs)")
        
        while True:
            try:
                animal_choice = input(f"\nSelect animal (1-{len(animal_list)}): ").strip()
                animal_idx = int(animal_choice) - 1
                if 0 <= animal_idx < len(animal_list):
                    animal_name = animal_list[animal_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(animal_list)}")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid number or Ctrl+C to exit")
        
        print(f"\nSelected: {animal_name}")
        
        try:
            # Analyze tracked ROI traces
            print("\n" + "=" * 60)
            print("ANALYZING TRACKED ROI LONGITUDINAL TRACES")
            print("=" * 60)
            
            analyze_tracked_rois_for_animal(
                data_path=data_path,
                animal_name=animal_name,
                save_path=save_path
            )
            
            print(f"\nAnalysis completed for {animal_name}")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please check the available animals above.")
    
    else:
        # All animals analysis mode
        print("=" * 60)
        print("ALL ANIMALS TRACKED ROI ANALYSIS MODE")
        print("=" * 60)
        print("This will process all animals that have tracked ROIs.")
        
        # Confirm before proceeding
        while True:
            try:
                confirm = input("\nProceed with analysis of all animals with tracked ROIs? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
                elif confirm in ['n', 'no']:
                    print("Analysis cancelled.")
                    exit()
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\nAnalysis cancelled.")
                exit()
        
        # Run analysis for all animals with tracked ROIs
        print("\n" + "=" * 60)
        print("RUNNING FULL TRACKED ROI ANALYSIS")
        print("=" * 60)
        analyze_all_animals_tracked_rois(data_path, save_path=save_path)
