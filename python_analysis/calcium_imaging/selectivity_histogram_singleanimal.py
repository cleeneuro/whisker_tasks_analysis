"""
Calculates selectivity indices for all ROIs for a single animal
User can input if they want a single session or all sessions for that animal
Plots selectivity histograms across learning phases pre, early, late, post

User must set data and save path for plots in the main function

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import pandas as pd

# Import shared functions
from data_loader import load_processed_data, list_available_animals_and_sessions
from calculate_selectivity import analyze_single_animal_session

# Add the data_processing directory to Python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)
from set_paths import processed_data_path, save_figures_path  # noqa: E402

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
        status_type (str): Type of status to filter by ('learner', 'cheater', 'sex', etc.)
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
        
        # Special handling for intrinsic imaging results
        if status_type == 'intrinsic_imaging_result':
            if animal_status == 'c1_c2':
                animals_by_status['c1_c2'].append(animal_name)
            elif animal_status != 'unknown':
                animals_by_status['other'].append(animal_name)
        else:
            # Standard handling for other status types
            if animal_status in status_values:
                animals_by_status[animal_status].append(animal_name)
    
    return animals_by_status

def plot_selectivity_histogram(selectivity_indices, save_path=None, figsize=(10, 6), 
                               animal_name=None, session_label=None):
    """
    Plot histogram of selectivity indices.
    
    Args:
        selectivity_indices (dict): Selectivity index for each ROI
        save_path (str, optional): Path to save plot
        figsize (tuple): Figure size
        animal_name (str, optional): Animal name for title
        session_label (str, optional): Session label for title
    """
    selectivities = [data['selectivity'] for data in selectivity_indices.values() 
                    if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    
    plt.figure(figsize=figsize)
    plt.hist(selectivities, bins=10, alpha=0.7, edgecolor='black')
    plt.xlim(-1, 1)
    plt.xlabel('Selectivity Index')
    plt.ylabel('Number of ROIs')
    
    # Create title with animal name and session if provided
    if animal_name and session_label:
        plt.title(f'Distribution of Selectivity Indices - {animal_name} {session_label}\n(CS+ - CS-) / (CS+ + CS-)')
    else:
        plt.title('Distribution of Selectivity Indices\n(CS+ - CS-) / (CS+ + CS-)')
    
    plt.legend()
    plt.grid(False)
    
    # Add statistics
    plt.text(0.05, 0.95, f'N ROIs: {len(selectivities)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        # Create folder structure: save_path/animal_name/
        if animal_name:
            plot_save_path = os.path.join(save_path, animal_name)
        else:
            plot_save_path = save_path
        os.makedirs(plot_save_path, exist_ok=True)
        
        # Create unique filename with animal name and session label
        if animal_name and session_label:
            filename = f'selectivity_histogram_{animal_name}_{session_label}.png'
        elif animal_name:
            filename = f'selectivity_histogram_{animal_name}.png'
        else:
            filename = 'selectivity_histogram.png'
            
        plt.savefig(os.path.join(plot_save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved selectivity histogram to {plot_save_path}")
    
    plt.show()

def plot_combined_selectivity_histograms(data_path, animal_name, save_path=None, 
                                        figsize=(20, 5), selectivity_method='peak'):
    """
    Plot selectivity histograms for all 4 phases (Pre, Early training, Late training, Post) 
    in a single figure with 4 subplots.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        save_path (str, optional): Path to save the combined plot
        figsize (tuple): Figure size for the combined plot
        selectivity_method (str): Method for selectivity calculation ('peak' or 'mean')
    """
    # Load data
    all_data = load_processed_data(data_path)
    
    if animal_name not in all_data:
        available_animals = list(all_data.keys())
        raise ValueError(f"Animal '{animal_name}' not found. Available animals: {available_animals}")
    
    animal_data = all_data[animal_name]
    sessions = animal_data['sessions']
    
    # Get all session labels and sort them
    session_labels = [session_data['label'] for session_data in sessions.values()]
    session_labels.sort()
    
    print(f"Available sessions for {animal_name}: {session_labels}")
    
    # Use session_types structure to properly categorize sessions
    session_types = animal_data['session_types']
    
    phase_mapping = {
        'Pre': None,
        'Early training': None, 
        'Late training': None,
        'Post': None
    }
    
    # Get Pre session
    if session_types['pre_session'] is not None:
        pre_session_idx = session_types['pre_session']
        phase_mapping['Pre'] = sessions[pre_session_idx]['label']
    
    # Get Post session
    if session_types['post_session'] is not None:
        post_session_idx = session_types['post_session']
        phase_mapping['Post'] = sessions[post_session_idx]['label']
    
    # Sort training sessions by day number (extract from Day1, Day2, etc.)
    training_sessions = []
    for session_idx in session_types['training_sessions']:
        session_label = sessions[session_idx]['label']
        # Extract day number from label (e.g., Day1 -> 1, Day10 -> 10)
        if session_label.startswith('Day'):
            try:
                day_num = int(session_label[3:])  # Extract number after 'Day'
                training_sessions.append((day_num, session_label))
            except ValueError:
                # If we can't parse the day number, skip this session
                continue
    
    # Sort by day number
    training_sessions.sort(key=lambda x: x[0])
    
    # Assign Early (Day 1-2) and Late (Day 3+) training
    for day_num, session_label in training_sessions:
        if day_num <= 2:  # Early training: Day 1 and Day 2
            if phase_mapping['Early training'] is None:
                phase_mapping['Early training'] = session_label
        else:  # Late training: Day 3 and beyond
            if phase_mapping['Late training'] is None:
                phase_mapping['Late training'] = session_label
            # If we have multiple late training sessions, keep the last one
            phase_mapping['Late training'] = session_label
    
    print(f"Phase mapping: {phase_mapping}")
    
    # Create figure with 4 subplots in horizontal layout (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f'Selectivity Indices (all ROIs) - {animal_name}', fontsize=16, fontweight='bold')
    
    # axes is already a 1D array for horizontal layout
    axes_flat = axes
    phase_names = ['Pre', 'Early training', 'Late training', 'Post']
    
    for i, (phase_name, session_label) in enumerate(phase_mapping.items()):
        ax = axes_flat[i]
        
        if session_label is None:
            # No data for this phase
            ax.text(0.5, 0.5, f'No {phase_name}\ndata available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(phase_name, fontsize=14, fontweight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            continue
        
        # Analyze this session
        try:
            results = analyze_single_animal_session(data_path, animal_name, session_label, 
                                                  save_path=None, generate_traces=False, 
                                                  generate_histograms=False,
                                                  selectivity_method=selectivity_method)
            selectivity_indices = results['selectivity_indices']
            
            # Extract selectivities
            selectivities = [data['selectivity'] for data in selectivity_indices.values() 
                           if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
            
            if len(selectivities) > 0:
                # Plot histogram with fixed bin width of 0.2 (2/10) and grey color
                bin_edges = np.arange(-1, 1.2, 0.2)  # Creates bins from -1 to 1 with width 0.2
                ax.hist(selectivities, bins=bin_edges, alpha=0.7, edgecolor='black', color='grey')
                ax.set_xlim(-1, 1)
                ax.set_xlabel('Selectivity Index')
                ax.set_ylabel('Number of ROIs')
                ax.set_title(f'{session_label}', fontsize=14, fontweight='bold')
                ax.grid(False)
                
                # Add statistics
                ax.text(0.05, 0.95, f'N ROIs: {len(selectivities)}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add mean of absolute values
                mean_abs_sel = np.mean(np.abs(selectivities))
                ax.text(0.05, 0.85, f'Mean |SI|: {mean_abs_sel:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'No valid\nselectivity data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{session_label}', fontsize=14, fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1)
                
        except Exception as e:
            print(f"Error analyzing {phase_name} ({session_label}): {e}")
            ax.text(0.5, 0.5, f'Error loading\n{session_label}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{session_label}', fontsize=14, fontweight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the combined plot
    if save_path:
        # Create folder structure: save_path/animal_name/
        plot_save_path = os.path.join(save_path, animal_name)
        os.makedirs(plot_save_path, exist_ok=True)
        filename = f'combined_selectivity_histograms_{animal_name}.png'
        plt.savefig(os.path.join(plot_save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved combined selectivity histograms to {plot_save_path}")
    
    plt.show()

def print_selectivity_indices(selectivity_indices, animal_name=None, session_label=None, sort_by='selectivity'):
    """
    Print all selectivity indices in a formatted table.
    
    Args:
        selectivity_indices (dict): Selectivity index for each ROI
        animal_name (str, optional): Animal name for display
        session_label (str, optional): Session label for display
        sort_by (str): How to sort the table - 'selectivity' (highest to lowest) or 'roi' (by ROI number)
    """
    if animal_name and session_label:
        print(f"\nSelectivity Indices for {animal_name} - {session_label}")
    else:
        print("\nSelectivity Indices")
    print("=" * 60)
    print(f"{'ROI':<6} {'Selectivity':<12} {'CS+ Response':<12} {'CS- Response':<12}")
    print("-" * 60)
    
    # Sort ROIs based on the specified method
    if sort_by == 'selectivity':
        sorted_rois = sorted(selectivity_indices.items(), key=lambda x: x[1]['selectivity'], reverse=True)
        print("(Sorted by selectivity index, highest to lowest)")
    elif sort_by == 'roi':
        sorted_rois = sorted(selectivity_indices.items(), key=lambda x: x[0])
        print("(Sorted by ROI number)")
    else:
        sorted_rois = list(selectivity_indices.items())
        print("(No sorting)")
    
    for roi_idx, data in sorted_rois:
        selectivity = data['selectivity']
        cs_plus_response = data['cs_plus_response']
        cs_minus_response = data['cs_minus_response']
        
        # ROI numbering: roi_idx is 0-based in the data, so roi_idx+1 gives us 1-based numbering
        print(f"{roi_idx+1:<6} {selectivity:<12.4f} {cs_plus_response:<12.4f} {cs_minus_response:<12.4f}")
    
    # Print summary statistics
    selectivities = [data['selectivity'] for data in selectivity_indices.values() 
                    if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    cs_plus_responses = [data['cs_plus_response'] for data in selectivity_indices.values() 
                        if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    cs_minus_responses = [data['cs_minus_response'] for data in selectivity_indices.values() 
                         if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    
    print("-" * 60)
    print(f"{'Mean':<6} {np.mean(selectivities):<12.4f} {np.mean(cs_plus_responses):<12.4f} {np.mean(cs_minus_responses):<12.4f}")
    print(f"{'Std':<6} {np.std(selectivities):<12.4f} {np.std(cs_plus_responses):<12.4f} {np.std(cs_minus_responses):<12.4f}")
    print(f"{'Min':<6} {np.min(selectivities):<12.4f} {np.min(cs_plus_responses):<12.4f} {np.min(cs_minus_responses):<12.4f}")
    print(f"{'Max':<6} {np.max(selectivities):<12.4f} {np.max(cs_plus_responses):<12.4f} {np.max(cs_minus_responses):<12.4f}")
    print("=" * 60)

def generate_histogram_single_animal(data_path, animal_name, session_label, save_path=None,
                                     selectivity_method='peak', combined_phases=False):
    """
    Generate selectivity histogram for a single animal and session.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Label of the session to analyze (e.g., 'Pre', 'Day1', 'Post')
        save_path (str, optional): Path to save plots
        selectivity_method (str): Method for selectivity calculation ('peak' or 'mean')
        combined_phases (bool): Whether to generate combined 4-phase histogram
        
    Returns:
        dict: Results containing selectivity_indices
    """
    if combined_phases:
        # Generate combined 4-phase histogram
        plot_combined_selectivity_histograms(
            data_path=data_path,
            animal_name=animal_name,
            save_path=save_path,
            selectivity_method=selectivity_method
        )
        return None
    else:
        # Generate single session histogram
        results = analyze_single_animal_session(
            data_path=data_path,
            animal_name=animal_name,
            session_label=session_label,
            save_path=None,
            generate_traces=False,
            generate_histograms=False,
            selectivity_method=selectivity_method
        )
        
        selectivity_indices = results['selectivity_indices']
        
        # Print selectivity indices and statistics
        print_selectivity_indices(selectivity_indices, animal_name, session_label, sort_by='selectivity')
        print_selectivity_indices(selectivity_indices, animal_name, session_label, sort_by='roi')
        print_selectivity_statistics(selectivity_indices, animal_name, session_label)
        
        # Plot histogram
        plot_selectivity_histogram(
            selectivity_indices,
            save_path=save_path,
            animal_name=animal_name,
            session_label=session_label
        )
        
        return results

def print_selectivity_statistics(selectivity_indices, animal_name=None, session_label=None):
    """
    Print detailed statistics for selectivity indices.
    
    Args:
        selectivity_indices (dict): Selectivity index for each ROI
        animal_name (str, optional): Animal name for display
        session_label (str, optional): Session label for display
    """
    if animal_name and session_label:
        print(f"\nSelectivity Statistics for {animal_name} - {session_label}")
    else:
        print("\nSelectivity Statistics")
    print("=" * 60)
    
    selectivities = [data['selectivity'] for data in selectivity_indices.values() 
                    if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    cs_plus_responses = [data['cs_plus_response'] for data in selectivity_indices.values() 
                        if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    cs_minus_responses = [data['cs_minus_response'] for data in selectivity_indices.values() 
                         if not np.isnan(data['selectivity']) and np.isfinite(data['selectivity'])]
    
    if len(selectivities) > 0:
        print(f"Total ROIs: {len(selectivities)}")
        print(f"Selectivity - Mean: {np.mean(selectivities):.4f} ± {np.std(selectivities):.4f}")
        print(f"Selectivity - Median: {np.median(selectivities):.4f}")
        print(f"Selectivity - Range: [{np.min(selectivities):.4f}, {np.max(selectivities):.4f}]")
        print(f"Mean |Selectivity|: {np.mean(np.abs(selectivities)):.4f}")
        print(f"CS+ Response - Mean: {np.mean(cs_plus_responses):.4f} ± {np.std(cs_plus_responses):.4f}")
        print(f"CS- Response - Mean: {np.mean(cs_minus_responses):.4f} ± {np.std(cs_minus_responses):.4f}")
    else:
        print("No valid selectivity data found")
    print("=" * 60)

if __name__ == "__main__":
    # Set paths
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'selectivity_histograms/single_animal')
    
    # Example usage
    print("=" * 60)
    print("SINGLE ANIMAL SELECTIVITY HISTOGRAM ANALYSIS")
    print("=" * 60)
    
    # Choose what to analyze
    print("\nChoose analysis type:")
    print("1. Single session histogram")
    print("2. Combined 4-phase histogram (Pre, Early, Late, Post)")
    
    while True:
        try:
            analysis_choice = input("\nEnter your choice (1 or 2): ").strip()
            if analysis_choice in ['1', '2']:
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
    
    combined_phases = (analysis_choice == '2')
    
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
    
    if not combined_phases:
        # Let user select session for single session analysis
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
    else:
        session_label = None
        print(f"\nSelected: {animal_name} - All phases")
    
    # Generate histogram for selected animal
    try:
        results = generate_histogram_single_animal(
            data_path=data_path,
            animal_name=animal_name,
            session_label=session_label,
            save_path=save_path,
            selectivity_method='peak',  # You can change this to 'mean' if preferred
            combined_phases=combined_phases
        )
        
        if results:
            print(f"\nAnalysis completed for {animal_name} - {session_label}")
        else:
            print(f"\nCombined analysis completed for {animal_name}")
        
    except ValueError as e:
        print(f"Error: {e}")

