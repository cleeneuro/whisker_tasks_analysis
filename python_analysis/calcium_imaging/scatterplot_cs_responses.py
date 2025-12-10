"""
Creates scatterplots of CS+ vs CS- responses for each learning phase.

4 different metrics are plotted:
- Baseline-Corrected Peak Amplitude
- Mean dF/F
- Baseline-Corrected Mean dF/F
- Area Under Curve

User can choose to plot all animals, a single animal, or a group of animals.

=============================================================================
For groups of animals, the user can select to group them based on 
- cheater vs non-cheater
- learner vs non-learner
- intrinsic imaging result (c1_c2 vs other)

GROUPING INFO IS SET IN data_processing/animal_metadata.csv
==============================================================================


"""
# %%
# %load_ext autoreload
# %autoreload 2

import re
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
import pandas as pd
import os
from matplotlib.collections import PolyCollection
import pickle
import sys

# Add the data_processing directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)
from learning_phase_mapping import day_label_to_learning_phase, convert_session_labels_to_phases, get_ordered_phase_labels  # noqa: E402
from set_paths import processed_data_path, save_figures_path  # noqa: E402
from data_loader import load_processed_data  # noqa: E402

# %%

# Set the data path
data_path = processed_data_path

# Load the data
all_animal_data = load_processed_data(data_path)

def load_animal_metadata(data_path):
    """
    Load animal metadata from the CSV file.
    This metadata includes learner/cheater status, sex, mouse line, etc.
    
    Args:
        data_path (str): Path to the pickle file containing processed data (used to get script directory)
        
    Returns:
        dict: Dictionary with animal names as keys and metadata as values
    """
    # Get the directory of the data processing module to find the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'data_processing', 'animal_metadata.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata CSV not found at {csv_path}")
        # Fallback: try to get animals from processed data with unknown metadata
        all_data = load_processed_data(data_path)
        metadata = {}
        for animal_name in all_data.keys():
            metadata[animal_name] = {
                'learner': 'unknown',
                'cheater': 'unknown',
                'sex': 'unknown',
                'mouse_line': 'unknown',
                'intrinsic_imaging_result': 'unknown'
            }
        return metadata
    
    # Load metadata from CSV
    df = pd.read_csv(csv_path)
    metadata = {}
    
    for _, row in df.iterrows():
        animal_id = row['animal_id']
        metadata[animal_id] = {
            'learner': row.get('learner', 'unknown'),
            'cheater': row.get('cheater', 'unknown'),
            'sex': row.get('sex', 'unknown'),
            'mouse_line': row.get('mouse_line', 'unknown'),
            'intrinsic_imaging_result': row.get('intrinsic_imaging_result', 'unknown')
        }
    
    return metadata

def get_animals_by_status(metadata, status_type='learner', status_values=None):
    """
    Get lists of animals based on their status.
    
    Args:
        metadata (dict): Animal metadata dictionary
        status_type (str): Type of status to filter by ('learner', 'cheater', 'intrinsic_imaging_result', etc.)
        status_values (list): List of status values to include (e.g., ['learner', 'non-learner'])
        
    Returns:
        dict: Dictionary with status values as keys and lists of animal names as values
    """
    if status_values is None:
        if status_type == 'learner':
            status_values = ['learner', 'non-learner']
        elif status_type == 'cheater':
            status_values = ['cheater', 'non-cheater']
        elif status_type == 'intrinsic_imaging_result':
            status_values = ['c1_c2', 'other']
        else:
            # Get all unique values for this status type
            status_values = list(set(animal_data[status_type] for animal_data in metadata.values()))
            status_values = [s for s in status_values if s != 'unknown']
    
    animals_by_status = {status: [] for status in status_values}
    
    for animal_name, animal_data in metadata.items():
        animal_status = animal_data.get(status_type, 'unknown')
        
        # Special handling for intrinsic_imaging_result
        if status_type == 'intrinsic_imaging_result':
            if animal_status == 'c1_c2':
                animals_by_status['c1_c2'].append(animal_name)
            elif animal_status not in ['unknown', 'n/a']:
                animals_by_status['other'].append(animal_name)
        # Special handling for learner (anything other than learner goes to non-learner)
        elif status_type == 'learner':
            if animal_status == 'learner':
                animals_by_status['learner'].append(animal_name)
            elif animal_status not in ['unknown']:
                animals_by_status['non-learner'].append(animal_name)
        else:
            if animal_status in status_values:
                animals_by_status[animal_status].append(animal_name)
    
    return animals_by_status

# %%

def computeROIResponseMetrics(stim_frames, dff, roi_indices, metric):
    """ 
    Loop through all trials of a single stim type and calculate response metrics
    for all ROIs indicated in roi_indices
    
    Metrics available:
    - 'peak_amplitude': Peak amplitude during stimulus window minus peak amplitude during baseline period (1s before stimulus)
    - 'mean_dFF': Mean df/f during 2s stimulus period  
    - 'baseline_corrected_mean': Mean df/f during stimulus minus mean df/f during 0.5s before stimulus
    - 'AUC': Area under the curve during stimulus period
    
    Args:
        stim_frames: List of (start_frame, stop_frame) tuples for stimulus trials
        dff: dF/F data array (ROIs x time)
        roi_indices: Array or list of ROI indices (0-based)
        metric: String specifying which metric to calculate
    """

    stim_responses = []

    for roi_idx in roi_indices:
        # Safety check to prevent index errors
        if roi_idx >= dff.shape[0]:
            print(f"    WARNING: ROI index {roi_idx} is out of bounds for dff array with {dff.shape[0]} ROIs. Skipping this ROI.")
            continue
   
        # Calculate response for each trial
        trial_responses = []
    
        for start_frame, stop_frame in stim_frames:
            # Calculate baseline (30 frames = 1 second before trial start)
            baseline_start = max(0, start_frame - 30)  # Ensure we don't go below 0
            baseline_end = start_frame
            
            # Get baseline and trial data
            baseline_dff = dff[roi_idx, baseline_start:baseline_end]
            trial_dff = dff[roi_idx, start_frame:stop_frame]
            
            # Calculate baseline average
            baseline_avg = np.mean(baseline_dff)
            
            # Calculate response metric based on selected type
            if metric == 'peak_amplitude':
                # Peak amplitude during stimulus window, corrected for baseline peak
                stimulus_peak = np.max(trial_dff) - np.min(trial_dff)
                baseline_peak = np.max(baseline_dff) - np.min(baseline_dff)
                response = stimulus_peak - baseline_peak
                
            elif metric == 'mean_dFF':
                # Mean df/f during stimulus period
                response = np.mean(trial_dff)
                
            elif metric == 'baseline_corrected_mean':
                # Mean df/f during stimulus minus baseline mean
                response = np.mean(trial_dff) - baseline_avg
                
            elif metric == 'AUC':
                # Area under the curve during stimulus (baseline corrected)
                baseline_corrected_dff = trial_dff - baseline_avg
                response = np.trapz(baseline_corrected_dff)
                
            else:
                raise ValueError(f"Unknown metric: {metric}. Available: 'peak_amplitude', 'mean_dFF', 'baseline_corrected_mean', 'AUC'")
            
            trial_responses.append(response)

        # Average across trials for this ROI
        stim_responses.append(np.mean(trial_responses))
            
    return stim_responses

# %%

def prepare_scatterplot_data(all_animal_data, metric='peak_amplitude', single_animal=None, animal_list=None):
    """
    Prepare CS+ and CS- response data for scatterplot visualization
    
    Args:
        all_animal_data: Dictionary containing all animal data
        metric: Response metric to calculate ('peak_amplitude', 'mean_dFF', 'baseline_corrected_mean', 'AUC')
        single_animal: If specified, only process this animal. If None, process all animals.
        animal_list: If specified, only process animals in this list (for pooling by metadata)
        
    Returns:
        Dictionary with learning phase labels as keys and arrays of (CS+, CS-) pairs as values
    """
    
    print(f'Preparing scatterplot data using {metric}...')
    
    if single_animal:
        if single_animal not in all_animal_data:
            available_animals = list(all_animal_data.keys())
            raise ValueError(f"Animal '{single_animal}' not found. Available animals: {available_animals}")
        animals_to_process = {single_animal: all_animal_data[single_animal]}
        print(f'Processing single animal: {single_animal}')
    elif animal_list:
        # Filter animals based on provided list (for pooling by metadata)
        animals_to_process = {}
        missing_animals = []
        for animal in animal_list:
            if animal in all_animal_data:
                animals_to_process[animal] = all_animal_data[animal]
            else:
                missing_animals.append(animal)
        
        if missing_animals:
            print(f'Warning: Animals not found in data: {missing_animals}')
        
        print(f'Processing filtered animals: {list(animals_to_process.keys())}')
    else:
        animals_to_process = all_animal_data
        print(f'Processing all animals: {list(all_animal_data.keys())}')

    # Initialize scatterplot data with proper learning phase labels
    standard_phases = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    scatterplot_data = {phase: [] for phase in standard_phases}
    
    for animal, animal_data in animals_to_process.items():
        print(f'Processing animal: {animal}')

        # Get learning phase mapping for this animal
        phase_mapping = convert_session_labels_to_phases(animal_data)
        ordered_phases = get_ordered_phase_labels(phase_mapping)
        
        print(f'  Learning phases for {animal}: {ordered_phases}')

        sessions = animal_data['sessions']
        for session_idx, session_data in sessions.items():
            session_label = session_data['label']
            
            # Convert session label to learning phase
            phase_label = day_label_to_learning_phase(session_label)
            
            # Skip if this phase is not in our standard phases
            if phase_label not in scatterplot_data:
                print(f'  Skipping unknown phase: {phase_label} (from {session_label})')
                continue

            # Use all ROIs (0-based indexing)
            num_rois = session_data['dff'].shape[0]
            all_roi_indices = np.arange(num_rois)  # 0-based indexing

            # Get data for session
            cs_plus_frames = session_data['cs_plus_frames']
            cs_minus_frames = session_data['cs_minus_frames']
            dff = session_data['dff']
            
            print(f'  {session_label} -> {phase_label}: {len(cs_plus_frames)} CS+ trials, {len(cs_minus_frames)} CS- trials, {num_rois} ROIs')
            
            # Calculate response for each ROI
            cs_plus_responses = computeROIResponseMetrics(cs_plus_frames, dff, all_roi_indices, metric=metric)
            cs_minus_responses = computeROIResponseMetrics(cs_minus_frames, dff, all_roi_indices, metric=metric)
            
            # Create (CS+, CS-) pairs for each ROI
            session_pairs = list(zip(cs_plus_responses, cs_minus_responses))
            scatterplot_data[phase_label].extend(session_pairs)

    # Convert lists to numpy arrays
    for label in scatterplot_data:
        scatterplot_data[label] = np.array(scatterplot_data[label])
        print(f'{label}: {scatterplot_data[label].shape[0]} ROI pairs')
        
    return scatterplot_data

# %%

def make_cs_response_scatterplot(scatterplot_data, metric='peak_amplitude', fig_savepath=None, animal_names=None, pool_title=None, grouped_data=None):
    """
    Create scatterplots of CS+ vs CS- responses for each learning phase
    
    Args:
        scatterplot_data: Dictionary with learning phase data (for single group)
        metric: Response metric used
        fig_savepath: Path to save figure
        animal_names: String of animal names for filename
        pool_title: Additional title for pooled data (e.g., "Learner vs Non-learner")
        grouped_data: Dictionary with group names as keys and scatterplot_data as values (for multi-group plots)
    """
    
    # Define axis limits and titles based on metric
    if metric == 'peak_amplitude':
        base_title = 'Baseline-Corrected Peak Amplitude per ROI: CS+ vs CS-'
        axis_lim = [-0.05, 0.8]
        xlabel = 'CS+ Baseline-Corrected Peak Amplitude'
        ylabel = 'CS- Baseline-Corrected Peak Amplitude'
    elif metric == 'mean_dFF':
        base_title = 'Mean dF/F per ROI: CS+ vs CS-'
        axis_lim = [-0.03, 0.4]
        xlabel = 'CS+ Mean dF/F'
        ylabel = 'CS- Mean dF/F'
    elif metric == 'baseline_corrected_mean':
        base_title = 'Baseline-Corrected Mean dF/F per ROI: CS+ vs CS-'
        axis_lim = [-0.05, 0.3]
        xlabel = 'CS+ Baseline-Corrected Mean dF/F'
        ylabel = 'CS- Baseline-Corrected Mean dF/F'
    elif metric == 'AUC':
        base_title = 'Area Under Curve per ROI: CS+ vs CS-'
        axis_lim = [-3, 17.5]
        xlabel = 'CS+ AUC'
        ylabel = 'CS- AUC'
    else:
        base_title = f'{metric} per ROI: CS+ vs CS-'
        axis_lim = [-1, 1]  # Default
        xlabel = f'CS+ {metric}'
        ylabel = f'CS- {metric}'
    
    # Add pool title if provided
    if pool_title:
        title = f'{base_title}\n{pool_title}'
    else:
        title = base_title

    # Create identity line for reference
    x = np.linspace(axis_lim[0], axis_lim[1], 100)
    y = x
    identity_line = (x, y)

    # Create figure with 1x4 subplots (single row)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, y=1.05, fontsize=16)
    
    # Define the layout of labels in the subplots (Pre, Early Learning, Late Learning, Post)
    subplot_layout = {
        'Pre': (ax1, 'Pre'),
        'Early Learning': (ax2, 'Early Learning'),
        'Late Learning': (ax3, 'Late Learning'),
        'Post': (ax4, 'Post')
    }
    
    # Define colors for different groups
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create scatterplots for each learning phase
    for label, (ax, phase_title) in subplot_layout.items():
        if grouped_data:
            # Multi-group plotting with color coding
            legend_handles = []
            total_points = 0
            
            for group_idx, (group_name, group_data) in enumerate(grouped_data.items()):
                if group_data[label].size > 0:
                    color = colors[group_idx % len(colors)]
                    scatter = ax.scatter(group_data[label][:, 0], group_data[label][:, 1], 
                                       alpha=0.6, edgecolors='black', s=20, color=color, label=group_name)
                    legend_handles.append(scatter)
                    total_points += group_data[label].shape[0]
            
            if total_points > 0:
                ax.plot(*identity_line, color='black', linestyle='--', linewidth=1)
                ax.set_title(phase_title, fontsize=14, fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_aspect('equal', adjustable='box')
                
                # Remove the top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add text with the number of data points
                ax.text(0.95, 0.05, f'n={total_points} ROIs', transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=10, color='grey')
                
                # Add legend to the last subplot only
                if label == 'Post' and legend_handles:
                    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
            else:
                # No data for this phase
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(phase_title, fontsize=14, fontweight='bold')
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_aspect('equal', adjustable='box')
        
        else:
            # Single group plotting (original logic)
            if scatterplot_data[label].size > 0:  # Only create plot if there's data
                ax.scatter(scatterplot_data[label][:, 0], scatterplot_data[label][:, 1], 
                          alpha=0.3, edgecolors='black', s=20)
                ax.plot(*identity_line, color='black', linestyle='--', linewidth=1)
                ax.set_title(phase_title, fontsize=14, fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_aspect('equal', adjustable='box')
                
                # Remove the top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add text with the number of data points
                num_points = scatterplot_data[label].shape[0]
                ax.text(0.95, 0.05, f'n={num_points} ROIs', transform=ax.transAxes, 
                       ha='right', va='bottom', fontsize=10, color='grey')
            else:
                # No data for this phase
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(phase_title, fontsize=14, fontweight='bold')
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    if fig_savepath is not None:
        save_path = fig_savepath
        os.makedirs(save_path, exist_ok=True)
        
        # Create filename with animal names
        if animal_names:
            filename = f"scatterplot_{metric}_{animal_names}.png"
        else:
            filename = f"scatterplot_{metric}.png"
            
        plt.savefig(f"{save_path}/{filename}", bbox_inches='tight', dpi=300)
        print(f'Saved scatterplot using {metric} to {save_path}/{filename}')
    else:
        print(f'Scatterplot not saved - no fig_savepath provided')
    
    plt.show()

# %%

def get_user_preferences(all_animal_data):
    """
    Ask user for their analysis preferences
    
    Args:
        all_animal_data: Dictionary containing all animal data
    
    Returns:
        tuple: (selected_metrics, selected_animal, pooling_config)
    """
    print("\n" + "=" * 60)
    print("CS RESPONSE SCATTERPLOT ANALYSIS")
    print("=" * 60)
    
    # Show available data - use wrapper function that doesn't require data_path
    def print_animal_data(all_data):
        """Print all available animals and sessions from loaded data"""
        animal_sessions = {}
        for animal_name, animal_data in all_data.items():
            sessions = animal_data['sessions']
            session_labels = [session_data['label'] for session_data in sessions.values()]
            animal_sessions[animal_name] = session_labels
        
        print("Available animals and sessions:")
        print("=" * 50)
        for animal_name, sessions in animal_sessions.items():
            print(f"{animal_name}:")
            for session in sessions:
                print(f"  - {session}")
            print()
    
    print_animal_data(all_animal_data)
    
    # Ask for metric choice
    print("\nWhich response metric would you like to analyze?")
    print("1. Baseline-corrected peak amplitude (stimulus peak - baseline peak)")
    print("2. Mean dF/F during 2s stimulus period")
    print("3. Baseline-corrected mean dF/F (stimulus - pre-stimulus baseline)")
    print("4. Area under the curve during stimulus period")
    print("5. All metrics")
    
    while True:
        try:
            metric_choice = input("\nEnter your choice (1-5): ").strip()
            if metric_choice == '1':
                selected_metrics = ['peak_amplitude']
                break
            elif metric_choice == '2':
                selected_metrics = ['mean_dFF']
                break
            elif metric_choice == '3':
                selected_metrics = ['baseline_corrected_mean']
                break
            elif metric_choice == '4':
                selected_metrics = ['AUC']
                break
            elif metric_choice == '5':
                selected_metrics = ['peak_amplitude', 'mean_dFF', 'baseline_corrected_mean', 'AUC']
                break
            else:
                print("Please enter 1, 2, 3, 4, or 5")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
    
    # Ask for animal choice
    print("\nWhich animals would you like to analyze?")
    available_animals = list(all_animal_data.keys())
    print("Available animals:", available_animals)
    print("1. Single animal")
    print("2. All animals pooled")
    print("3. Pool animals by metadata (cheater, learner, intrinsic imaging result)")
    
    pooling_config = None
    
    while True:
        try:
            animal_choice = input("\nEnter your choice (1-3): ").strip()
            if animal_choice == '1':
                # Let user select specific animal
                print("\nSelect animal:")
                for i, animal in enumerate(available_animals, 1):
                    print(f"{i}. {animal}")
                
                while True:
                    try:
                        animal_idx = int(input(f"\nEnter animal number (1-{len(available_animals)}): ").strip()) - 1
                        if 0 <= animal_idx < len(available_animals):
                            selected_animal = available_animals[animal_idx]
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(available_animals)}")
                    except ValueError:
                        print("Please enter a valid number")
                break
            elif animal_choice == '2':
                selected_animal = None  # Process all animals
                break
            elif animal_choice == '3':
                # Pool animals by metadata
                selected_animal = None
                
                print("\nSelect pooling criteria:")
                print("1. Cheater: cheater vs non-cheater")
                print("2. Learner: learner vs anything other than learner")
                print("3. Intrinsic imaging result FOV: 'c1_c2' vs anything else")
                
                while True:
                    try:
                        pool_choice = input("\nEnter pooling choice (1-3): ").strip()
                        if pool_choice == '1':
                            pooling_config = {
                                'status_type': 'cheater',
                                'status_values': ['cheater', 'non-cheater'],
                                'title': 'Cheater vs Non-cheater'
                            }
                            break
                        elif pool_choice == '2':
                            pooling_config = {
                                'status_type': 'learner',
                                'status_values': ['learner', 'non-learner'],
                                'title': 'Learner vs Non-learner'
                            }
                            break
                        elif pool_choice == '3':
                            pooling_config = {
                                'status_type': 'intrinsic_imaging_result',
                                'status_values': ['c1_c2', 'other'],
                                'title': 'C1/C2 FOV vs Other FOV'
                            }
                            break
                        else:
                            print("Please enter 1, 2, or 3")
                    except ValueError:
                        print("Please enter a valid number")
                break
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
    
    return selected_metrics, selected_animal, pooling_config

# %%

def main():
    """
    Main function to run the CS response scatterplot analysis
    """
    
    # Get user preferences
    selected_metrics, selected_animal, pooling_config = get_user_preferences(all_animal_data)
    
    # Handle pooling configuration
    if pooling_config:
        # Load metadata and get animals by status
        metadata = load_animal_metadata(data_path)
        animals_by_status = get_animals_by_status(
            metadata, 
            status_type=pooling_config['status_type'],
            status_values=pooling_config['status_values']
        )
        
        print(f"\nAnimals by status ({pooling_config['status_type']}):")
        for status, animals in animals_by_status.items():
            print(f"  {status}: {animals}")
        
        # Set up save path for pooled data
        animal_string = f"{pooling_config['status_type']}_comparison"
        save_path = os.path.join(save_figures_path, 'cs_response_scatterplots', animal_string)
        
        print(f"\nSave path: {save_path}")
        
        # Process each selected metric
        for metric in selected_metrics:
            print(f"\n" + "=" * 60)
            print(f"PROCESSING METRIC: {metric.upper()} - {pooling_config['title'].upper()}")
            print("=" * 60)
            
            # Prepare grouped data for all status groups
            grouped_data = {}
            for status, animal_list in animals_by_status.items():
                if animal_list:  # Only process if there are animals in this group
                    print(f"Processing {status} animals: {animal_list}")
                    scatterplot_data = prepare_scatterplot_data(
                        all_animal_data, 
                        metric=metric, 
                        animal_list=animal_list
                    )
                    grouped_data[status] = scatterplot_data
                else:
                    print(f"Skipping {status} - no animals found")
            
            # Create single scatterplot with color-coded groups
            if grouped_data:
                make_cs_response_scatterplot(
                    scatterplot_data=None,  # Not used when grouped_data is provided
                    metric=metric, 
                    fig_savepath=save_path, 
                    animal_names=animal_string,
                    pool_title=pooling_config['title'],
                    grouped_data=grouped_data
                )
    else:
        # Original logic for single animal or all animals
        if selected_animal:
            animal_string = selected_animal
            save_path = os.path.join(save_figures_path, 'cs_response_scatterplots', animal_string)
        else:
            animal_names = list(all_animal_data.keys())
            animal_string = 'pooled'
            save_path = os.path.join(save_figures_path, 'cs_response_scatterplots', animal_string)
        
        print(f"\nProcessing with the following settings:")
        print(f"Metrics: {selected_metrics}")
        print(f"Animal(s): {selected_animal if selected_animal else 'All animals'}")
        print(f"Save path: {save_path}")
        
        # Process each selected metric
        for metric in selected_metrics:
            print(f"\n" + "=" * 60)
            print(f"PROCESSING METRIC: {metric.upper()}")
            print("=" * 60)
            
            # Prepare data
            scatterplot_data = prepare_scatterplot_data(
                all_animal_data, 
                metric=metric, 
                single_animal=selected_animal
            )
            
            # Create scatterplot
            make_cs_response_scatterplot(
                scatterplot_data, 
                metric=metric, 
                fig_savepath=save_path, 
                animal_names=animal_string
            )
    
    print(f"\nAnalysis complete! Results saved to: {save_path}")

# %%

if __name__ == "__main__":
    main()

# %%
