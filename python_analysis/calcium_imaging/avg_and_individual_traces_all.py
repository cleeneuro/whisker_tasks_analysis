"""
Average a single ROI's df/f traces across all CS+ and CS- trials within a session.
Plots average traces for all ROIs but also overlay traces from individual trial.

Creates subplots with CS+ and CS- traces are separated into different plots.

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
from set_paths import processed_data_path, save_figures_path  # noqa: E402
from extract_trial_dff import extract_trial_dff  # noqa: E402
from data_loader import load_processed_data, list_available_animals_and_sessions, print_available_data  # noqa: E402

def plot_individual_trial_traces(cs_plus_trials, cs_minus_trials, time_axis, save_path=None, 
                                max_rois_per_figure=10, figsize=(16, 10), animal_name=None, session_label=None):
    """
    Plot individual trial traces for each ROI with CS+ and CS- side by side.
    Creates multiple figures if there are more ROIs than can fit on one figure.
    Each ROI gets two subplots: CS+ on the left, CS- on the right.
    
    Args:
        cs_plus_trials (np.ndarray): CS plus trial data (n_trials, n_rois, n_frames)
        cs_minus_trials (np.ndarray): CS minus trial data (n_trials, n_rois, n_frames)
        time_axis (np.ndarray): Time axis in seconds
        save_path (str, optional): Path to save plots
        max_rois_per_figure (int): Maximum number of ROIs per figure (reduced default for side-by-side)
        figsize (tuple): Figure size (increased default for side-by-side layout)
        animal_name (str, optional): Name of the animal for the title
        session_label (str, optional): Session label for the title
    """
    n_trials_plus, n_rois, n_frames = cs_plus_trials.shape
    n_trials_minus = cs_minus_trials.shape[0]
    
    print(f"Plotting individual trial traces for {n_rois} ROIs (CS+ and CS- side by side)")
    print(f"CS+ trials: {n_trials_plus}, CS- trials: {n_trials_minus}")
    
    # Calculate how many figures we need
    n_figures = int(np.ceil(n_rois / max_rois_per_figure))
    
    print(f"Creating {n_figures} figure(s) for {n_rois} ROIs ({max_rois_per_figure} ROIs per figure)")
    
    for fig_num in range(n_figures):
        # Determine ROIs for this figure
        start_idx = fig_num * max_rois_per_figure
        end_idx = min(start_idx + max_rois_per_figure, n_rois)
        figure_rois = list(range(start_idx, end_idx))
        n_rois_in_fig = len(figure_rois)
        
        # Create subplots - 2 columns per ROI (CS+ and CS-), arrange ROIs in rows
        # Each ROI gets 2 columns (CS+ left, CS- right)
        n_cols = 4  # 2 ROIs per row (2 ROIs × 2 plots each = 4 columns)
        n_rows = int(np.ceil(n_rois_in_fig / 2))  # 2 ROIs per row
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Add overall title if animal name and session label are provided
        if animal_name and session_label:
            fig.suptitle(f'{animal_name} - {session_label}', fontsize=16, fontweight='bold', y=0.98)
        elif animal_name:
            fig.suptitle(f'{animal_name}', fontsize=16, fontweight='bold', y=0.98)
        elif session_label:
            fig.suptitle(f'{session_label}', fontsize=16, fontweight='bold', y=0.98)
        
        # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for roi_num, roi_idx in enumerate(figure_rois):
            # Calculate subplot positions
            row = roi_num // 2  # Which row this ROI is in
            col_start = (roi_num % 2) * 2  # Starting column (0 or 2)
            
            # Get axes for CS+ (left) and CS- (right)
            ax_plus = axes[row, col_start] if n_cols > 1 else axes[row]
            ax_minus = axes[row, col_start + 1] if n_cols > 1 else axes[row]
            
            # Get individual trial data for this ROI
            cs_plus_roi_trials = cs_plus_trials[:, roi_idx, :]  # Shape: (n_trials, n_frames)
            cs_minus_roi_trials = cs_minus_trials[:, roi_idx, :]  # Shape: (n_trials, n_frames)
            
            # Add stimulus time shaded patch (behind all traces)
            for ax in [ax_plus, ax_minus]:
                ax.axvspan(xmin=0, xmax=2, color='pink', alpha=0.2, zorder=0)
            
            # add reward line for CS+ trials
            for ax in [ax_plus]:
                ax.axvline(x=2, color='k', linestyle='--', alpha=0.7, linewidth=1)
            
            # Plot CS+ trials (vectorized for speed)
            ax_plus.plot(time_axis, cs_plus_roi_trials.T, 
                        color='steelblue', alpha=0.6, linewidth=0.5)
            
            # Plot CS- trials (vectorized for speed)
            ax_minus.plot(time_axis, cs_minus_roi_trials.T, 
                         color='grey', alpha=0.6, linewidth=0.5)
            
            # Calculate and plot average traces
            cs_plus_mean = np.mean(cs_plus_roi_trials, axis=0)
            cs_minus_mean = np.mean(cs_minus_roi_trials, axis=0)
            
            # Plot thick average traces
            ax_plus.plot(time_axis, cs_plus_mean, 'b-', linewidth=3, label='CS+ (avg)', alpha=0.9)
            ax_minus.plot(time_axis, cs_minus_mean, 'k-', linewidth=3, label='CS- (avg)', alpha=0.9)
            
            # Set specific axis limits for both subplots
            for ax in [ax_plus, ax_minus]:
                ax.set_ylim(-0.5, 2.5)
                ax.set_xlim(-1.0, 4)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('dF/F')
            
            # Set titles
            ax_plus.set_title(f'ROI {roi_idx + 1} - CS+', fontsize=10, fontweight='bold')
            ax_minus.set_title(f'ROI {roi_idx + 1} - CS-', fontsize=10, fontweight='bold')
            
        
        # Hide unused subplots
        total_subplots = n_rows * n_cols
        used_subplots = n_rois_in_fig * 2  # Each ROI uses 2 subplots
        
        if total_subplots > used_subplots:
            axes_flat = axes.flatten() if axes.ndim > 1 else [axes]
            for i in range(used_subplots, total_subplots):
                if i < len(axes_flat):
                    axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            if n_figures == 1:
                filename = f'{animal_name}_{session_label}_traces.png'
            else:
                filename = f'{animal_name}_{session_label}_traces_{fig_num + 1}.png'
            plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches='tight')
            print(f"Saved individual trial traces plot {fig_num + 1}/{n_figures} to {save_path}")
        
        plt.show()
        plt.close()  # Close the figure to free memory

def analyze_individual_trial_traces(data_path, animal_name, session_label, save_path=None, 
                                   max_rois_to_plot=10):
    """
    Analyze and plot individual trial traces for a specific animal and session.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Label of the session to analyze (e.g., 'Pre', 'Day1', 'Post')
        save_path (str, optional): Path to save plots and results
        max_rois_to_plot (int): Maximum number of ROIs to plot per figure
        
    Returns:
        dict: Results containing trial data and time axis
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
    target_session_idx = None
    
    for session_idx, session_data in sessions.items():
        if session_data['label'] == session_label:
            target_session = session_data
            target_session_idx = session_idx
            break
    
    if target_session is None:
        available_sessions = [session_data['label'] for session_data in sessions.values()]
        raise ValueError(f"Session '{session_label}' not found for animal '{animal_name}'. Available sessions: {available_sessions}")
    
    print(f"Processing individual trial traces for {animal_name} - {session_label}...")
    
    # Get session data
    dff = target_session['dff']
    cs_plus_frames = target_session['cs_plus_frames']
    cs_minus_frames = target_session['cs_minus_frames']
    
    print(f"  CS+ trials: {len(cs_plus_frames)}")
    print(f"  CS- trials: {len(cs_minus_frames)}")
    print(f"  dF/F data shape: {dff.shape}")
    
    # Extract trial responses
    cs_plus_trials, time_axis = extract_trial_dff(dff, cs_plus_frames)
    cs_minus_trials, _ = extract_trial_dff(dff, cs_minus_frames)
    
    # Plot individual trial traces
    if save_path:
        # Create folder structure: /all_rois/animalname/
        animal_folder = os.path.join(save_path, "all_rois", animal_name)
        os.makedirs(animal_folder, exist_ok=True)
        session_save_path = animal_folder
    else:
        session_save_path = None
    
    plot_individual_trial_traces(cs_plus_trials, cs_minus_trials, time_axis, 
                                save_path=session_save_path, max_rois_per_figure=max_rois_to_plot,
                                animal_name=animal_name, session_label=session_label)
    
    print(f"Completed individual trial trace analysis for {animal_name} - {session_label}")
    
    # Return results for further analysis
    results = {
        'animal_name': animal_name,
        'session_label': session_label,
        'cs_plus_trials': cs_plus_trials,
        'cs_minus_trials': cs_minus_trials,
        'time_axis': time_axis,
        'n_rois': cs_plus_trials.shape[1],
        'n_trials_plus': cs_plus_trials.shape[0],
        'n_trials_minus': cs_minus_trials.shape[0]
    }
    
    return results

def analyze_all_animals_individual_trials(data_path, save_path=None, max_rois_to_plot=10):
    """
    Analyze and plot individual trial traces for all animals and all their sessions.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str, optional): Path to save plots and results
        max_rois_to_plot (int): Maximum number of ROIs to plot per figure
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Process each animal
    for animal_name, animal_data in all_data.items():
        print(f"\nProcessing {animal_name}...")
        
        # Process each session
        sessions = animal_data['sessions']
        for session_idx, session_data in sessions.items():
            session_label = session_data['label']
            print(f"  Processing {session_label}...")
            
            try:
                # Get session data
                dff = session_data['dff']
                cs_plus_frames = session_data['cs_plus_frames']
                cs_minus_frames = session_data['cs_minus_frames']
                
                print(f"    CS+ trials: {len(cs_plus_frames)}")
                print(f"    CS- trials: {len(cs_minus_frames)}")
                print(f"    dF/F data shape: {dff.shape}")
                
                # Extract trial responses
                cs_plus_trials, time_axis = extract_trial_dff(dff, cs_plus_frames)
                cs_minus_trials, _ = extract_trial_dff(dff, cs_minus_frames)
                
                # Plot individual trial traces
                if save_path:
                    # Create folder structure: /average_traces/individual_trial_overlay/all_rois/animalname/
                    animal_folder = os.path.join(save_path, "all_rois", animal_name)
                    os.makedirs(animal_folder, exist_ok=True)
                    session_save_path = animal_folder
                else:
                    session_save_path = None
                
                plot_individual_trial_traces(cs_plus_trials, cs_minus_trials, time_axis, 
                                            save_path=session_save_path, max_rois_per_figure=max_rois_to_plot,
                                            animal_name=animal_name, session_label=session_label)
                
                
            except Exception as e:
                print(f"    Error processing {animal_name} - {session_label}: {e}")
                continue

if __name__ == "__main__":
    # Set paths
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'average_traces','individual_trial_overlay')
    
    # List all available animals and sessions
    print("=" * 60)
    print("LISTING AVAILABLE DATA")
    print("=" * 60)
    print_available_data(data_path)
    
    # Ask user to choose analysis mode
    print("\n" + "=" * 60)
    print("ANALYSIS MODE SELECTION")
    print("=" * 60)
    print("Choose analysis mode:")
    print("1. Single animal/session analysis")
    print("2. All animals analysis (process all animals and sessions)")
    
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
    print(f"Selected mode: {'Single animal/session analysis' if analysis_mode == 'single' else 'All animals analysis'}")
    
    if analysis_mode == "single":
        # Single animal analysis mode
        print("=" * 60)
        print("SINGLE ANIMAL/SESSION ANALYSIS MODE")
        print("=" * 60)
        
        # Get available animals and sessions for user selection
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
        
        try:
            # Analyze individual trial traces
            print("\n" + "=" * 60)
            print("ANALYZING INDIVIDUAL TRIAL TRACES")
            print("=" * 60)
            
            results = analyze_individual_trial_traces(
                data_path=data_path,
                animal_name=animal_name,
                session_label=session_label,
                save_path=save_path,
                max_rois_to_plot=10
            )
            
            # Print summary
            print(f"\nAnalysis completed for {animal_name} - {session_label}")
            print(f"Number of ROIs: {results['n_rois']}")
            print(f"CS+ trials: {results['n_trials_plus']}")
            print(f"CS- trials: {results['n_trials_minus']}")
            print(f"Time axis length: {len(results['time_axis'])}")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please check the available animals and sessions above.")
    
    else:
        # All animals analysis mode
        print("=" * 60)
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
        
        # Run analysis for all animals and sessions
        print("\n" + "=" * 60)
        print("RUNNING FULL ANALYSIS")
        print("=" * 60)
        analyze_all_animals_individual_trials(data_path, save_path=save_path, max_rois_to_plot=10)
