"""
Average a single ROI's df/f traces across all CS+ and CS- trials within a session.
Plots average traces for all ROIs. One subplot for each ROI.

CS+ and CS- traces are overlaid on same plot
User can input if they want all animals for all sessions or a single animal for a single session
This does not include individual trial traces - you can find that in plot_avg_trace_with_inidivdual_traces.py

Data path and save path are set in set_paths.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import data loading functions
from data_loader import load_processed_data, list_available_animals_and_sessions
from extract_trial_dff import extract_trial_dff

# Add the data_processing directory to Python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)
from set_paths import processed_data_path, save_figures_path  # noqa: E402


def calculate_roi_averaged_traces(trial_responses, roi_indices=None):
    """
    Calculate averaged fluorescence traces for each ROI across trials.
    
    Args:
        trial_responses (np.ndarray): Array of shape (n_trials, n_rois, n_frames)
        roi_indices (np.ndarray, optional): Specific ROI indices to analyze
        
    Returns:
        dict: Dictionary containing mean traces and SEM for each ROI
    """
    n_trials, n_rois, n_frames = trial_responses.shape
    
    if roi_indices is None:
        roi_indices = np.arange(n_rois)
    
    roi_data = {}
    
    for roi_idx in roi_indices:
        # Extract trials for this ROI
        roi_trials = trial_responses[:, roi_idx, :]  # Shape: (n_trials, n_frames)
        
        # Calculate mean and SEM across trials
        mean_trace = np.mean(roi_trials, axis=0)
        sem_trace = np.std(roi_trials, axis=0) / np.sqrt(roi_trials.shape[0])
        
        roi_data[roi_idx] = {
            'mean_trace': mean_trace,
            'sem_trace': sem_trace,
            'n_trials': roi_trials.shape[0]
        }
    
    return roi_data

def plot_cs_traces_with_sem(cs_plus_data, cs_minus_data, time_axis, save_path=None, 
                           max_rois_per_figure=20, figsize=(12, 8), animal_name=None, session_label=None):
    """
    Plot CS plus and CS minus averaged traces with SEM error bars for each ROI.
    Creates multiple figures if there are more ROIs than can fit on one figure.
    
    Args:
        cs_plus_data (dict): CS plus data for each ROI
        cs_minus_data (dict): CS minus data for each ROI  
        time_axis (np.ndarray): Time axis in seconds
        save_path (str, optional): Base path to save plots
        max_rois_per_figure (int): Maximum number of ROIs per figure
        figsize (tuple): Figure size
        animal_name (str, optional): Animal name for folder structure
        session_label (str, optional): Session label for folder structure
    """
    roi_indices = list(cs_plus_data.keys())
    total_rois = len(roi_indices)
    
    # Calculate how many figures we need
    n_figures = int(np.ceil(total_rois / max_rois_per_figure))
    
    print(f"Creating {n_figures} figure(s) for {total_rois} ROIs ({max_rois_per_figure} ROIs per figure)")
    
    for fig_num in range(n_figures):
        # Determine ROIs for this figure
        start_idx = fig_num * max_rois_per_figure
        end_idx = min(start_idx + max_rois_per_figure, total_rois)
        figure_rois = roi_indices[start_idx:end_idx]
        n_rois_in_fig = len(figure_rois)
        
        # Create subplots
        n_cols = 4
        n_rows = int(np.ceil(n_rois_in_fig / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, roi_idx in enumerate(figure_rois):
            ax = axes_flat[i]
            
            # Get data for this ROI
            cs_plus_mean = cs_plus_data[roi_idx]['mean_trace']
            cs_plus_sem = cs_plus_data[roi_idx]['sem_trace']
            cs_minus_mean = cs_minus_data[roi_idx]['mean_trace']
            cs_minus_sem = cs_minus_data[roi_idx]['sem_trace']
            
            # Plot CS plus with shaded error bars
            ax.plot(time_axis, cs_plus_mean, 'b-', linewidth=2, label='CS+')
            ax.fill_between(time_axis, cs_plus_mean - cs_plus_sem, cs_plus_mean + cs_plus_sem, 
                           color='blue', alpha=0.3)
            
            # Plot CS minus with shaded error bars
            ax.plot(time_axis, cs_minus_mean, 'k-', linewidth=2, label='CS-')
            ax.fill_between(time_axis, cs_minus_mean - cs_minus_sem, cs_minus_mean + cs_minus_sem, 
                           color='black', alpha=0.3)
            
            # Add stimulus time shaded patch (behind all traces)
            ax.axvspan(xmin=0, xmax=2, color='pink', alpha=0.2, zorder=0)
            
            # Formatting
            ax.set_title(f'ROI {roi_idx + 1}', fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('dF/F')
            ax.grid(False)
            
            # Set specific axis limits as requested
            ax.set_ylim(-0.1, 0.5)
            ax.set_xlim(-1.0, 4)
        
        # Hide unused subplots
        for i in range(n_rois_in_fig, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            # Create folder structure: average_traces/all_rois/animal_name/
            if animal_name:
                plot_save_path = os.path.join(save_path, 'average_traces','all_rois', animal_name)
            else:
                plot_save_path = os.path.join(save_path, 'average_traces','all_rois')
            os.makedirs(plot_save_path, exist_ok=True)
            
            # Generate filename with animal name and session label, and figure number
            if animal_name and session_label:
                if n_figures == 1:
                    filename = f'{animal_name}_{session_label}_average_traces_with_sem.png'
                else:
                    filename = f'{animal_name}_{session_label}_average_traces_with_sem_{fig_num + 1}.png'
            else:
                # Fallback to generic names if animal_name or session_label not provided
                if n_figures == 1:
                    filename = 'cs_response_traces_with_sem.png'
                else:
                    filename = f'cs_response_traces_with_sem_figure_{fig_num + 1}.png'
            plt.savefig(os.path.join(plot_save_path, filename), dpi=300, bbox_inches='tight')
            print(f"Saved CS traces plot {fig_num + 1}/{n_figures} to {plot_save_path}")
        
        plt.show()

def generate_traces_single_animal(data_path, animal_name, session_label, save_path=None, 
                                  max_rois_to_plot=20):
    """
    Generate average traces for a single animal and session.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Label of the session to analyze (e.g., 'Pre', 'Day1', 'Post')
        save_path (str, optional): Path to save plots
        max_rois_to_plot (int): Maximum number of ROIs to plot
        
    Returns:
        dict: Results containing cs_plus_data, cs_minus_data, and time_axis
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    # Check if animal exists
    if animal_name not in all_data:
        available_animals = list(all_data.keys())
        raise ValueError(f"Animal '{animal_name}' not found. Available animals: {available_animals}")
    
    animal_data = all_data[animal_name]
    
    # Find the session
    sessions = animal_data['sessions']
    target_session = None
    
    for session_idx, session_data in sessions.items():
        if session_data['label'] == session_label:
            target_session = session_data
            break
    
    if target_session is None:
        available_sessions = [session_data['label'] for session_data in sessions.values()]
        raise ValueError(f"Session '{session_label}' not found for animal '{animal_name}'. Available sessions: {available_sessions}")
    
    print(f"Processing {animal_name} - {session_label}...")
    
    # Use all ROIs
    roi_indices = None
    
    # Get session data
    dff = target_session['dff']
    cs_plus_frames = target_session['cs_plus_frames']
    cs_minus_frames = target_session['cs_minus_frames']
    
    print(f"  CS+ trials: {len(cs_plus_frames)}")
    print(f"  CS- trials: {len(cs_minus_frames)}")
    print(f"  dF/F data shape: {dff.shape}")
    
    # Extract trial dff for plotting
    cs_plus_trials_raw, time_axis = extract_trial_dff(dff, cs_plus_frames)
    cs_minus_trials_raw, _ = extract_trial_dff(dff, cs_minus_frames)
    
    # Calculate ROI-averaged traces 
    cs_plus_data = calculate_roi_averaged_traces(cs_plus_trials_raw, roi_indices)
    cs_minus_data = calculate_roi_averaged_traces(cs_minus_trials_raw, roi_indices)
    
    # Plot traces
    plot_cs_traces_with_sem(cs_plus_data, cs_minus_data, time_axis, 
                           save_path=save_path, max_rois_per_figure=max_rois_to_plot,
                           animal_name=animal_name, session_label=session_label)
    
    print(f"Completed {animal_name} - {session_label}: {len(cs_plus_data)} ROIs analyzed")
    
    # Return results
    results = {
        'animal_name': animal_name,
        'session_label': session_label,
        'cs_plus_data': cs_plus_data,
        'cs_minus_data': cs_minus_data,
        'time_axis': time_axis,
        'roi_indices': roi_indices
    }
    
    return results

def generate_traces_all_animals(data_path, save_path=None, max_rois_to_plot=20):
    """
    Generate average traces for all animals and all sessions.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str, optional): Path to save plots
        max_rois_to_plot (int): Maximum number of ROIs to plot
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Process each animal
    for animal_name, animal_data in all_data.items():
        print(f"\nProcessing {animal_name}...")
        
        # Use all ROIs
        roi_indices = None
        
        # Process each session
        sessions = animal_data['sessions']
        for session_idx, session_data in sessions.items():
            session_label = session_data['label']
            print(f"  Processing {session_label}...")
            
            # Get session data
            dff = session_data['dff']
            cs_plus_frames = session_data['cs_plus_frames']
            cs_minus_frames = session_data['cs_minus_frames']
            
            # Extract raw trial responses (for plotting)
            cs_plus_trials_raw, time_axis = extract_trial_dff(dff, cs_plus_frames)
            cs_minus_trials_raw, _ = extract_trial_dff(dff, cs_minus_frames)
            
            # Calculate ROI-averaged traces (raw for plotting)
            cs_plus_data = calculate_roi_averaged_traces(cs_plus_trials_raw, roi_indices)
            cs_minus_data = calculate_roi_averaged_traces(cs_minus_trials_raw, roi_indices)
            
            # Plot traces
            plot_cs_traces_with_sem(cs_plus_data, cs_minus_data, time_axis, 
                                   save_path=save_path, max_rois_per_figure=max_rois_to_plot,
                                   animal_name=animal_name, session_label=session_label)
            
            print(f"  Completed {session_label}: {len(cs_plus_data)} ROIs analyzed")

if __name__ == "__main__":
    # Set paths
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'average_traces')
    
    # List available animals and sessions
    print("=" * 60)
    print("AVERAGE TRACES ANALYSIS")
    print("=" * 60)
    
    # Ask if user wants one animal or all animals
    print("\nChoose analysis mode:")
    print("1. Single animal")
    print("2. All animals")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == '1':
                analysis_mode = "single"
                break
            elif choice == '2':
                analysis_mode = "all"
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
    
    if analysis_mode == "single":
        # Single animal mode
        print("\n" + "=" * 60)
        print("SINGLE ANIMAL ANALYSIS MODE")
        print("=" * 60)
        
        # Get available animals and sessions
        animal_sessions = list_available_animals_and_sessions(data_path)
        
        # Let user select animal
        print("\nAvailable animals:")
        animal_list = list(animal_sessions.keys())
        for i, animal in enumerate(animal_list, 1):
            print(f"{i}. {animal}")
        
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
        
        # Let user select session
        sessions = animal_sessions[animal_name]
        print(f"\nAvailable sessions for {animal_name}:")
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session}")
        
        while True:
            try:
                session_choice = input(f"\nSelect session (1-{len(sessions)}): ").strip()
                session_idx = int(session_choice) - 1
                if 0 <= session_idx < len(sessions):
                    session_label = sessions[session_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(sessions)}")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid number or Ctrl+C to exit")
        
        print(f"\nSelected: {animal_name} - {session_label}")
        
        # Generate traces for selected animal and session
        try:
            results = generate_traces_single_animal(
                data_path=data_path,
                animal_name=animal_name,
                session_label=session_label,
                save_path=save_path,
                max_rois_to_plot=20
            )
            
            print(f"\nAnalysis completed for {animal_name} - {session_label}")
            print(f"Number of ROIs analyzed: {len(results['cs_plus_data'])}")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    else:
        # All animals mode
        print("\n" + "=" * 60)
        print("ALL ANIMALS ANALYSIS MODE")
        print("=" * 60)
        print("This will process all animals and all their sessions.")
        
        # Confirm before proceeding
        while True:
            try:
                confirm = input("\nProceed with analysis of all animals? (y/n): ").strip().lower()
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
        
        # Generate traces for all animals
        generate_traces_all_animals(
            data_path=data_path,
            save_path=save_path,
            max_rois_to_plot=20
        )
        
        print("\nAnalysis completed for all animals!")



