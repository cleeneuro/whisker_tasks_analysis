"""
Calculates selectivity indices for all ROIs 
Plots selectivity histograms across learning phases pre, early, late, post

Pools across all animals or across specific groups based on user input 

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

def pool_selectivity_data_by_status(data_path, status_type='learner', session_label=None, 
                                   status_values=None, selectivity_method='peak'):
    """
    Pool selectivity indices across mice based on their status for a specific session.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        status_type (str): Type of status to group by ('learner', 'cheater', 'sex', etc.)
        session_label (str): Specific session to analyze (e.g., 'Pre', 'Day1', 'Post')
        status_values (list): List of status values to include
        selectivity_method (str): Method for selectivity calculation ('peak' or 'mean')
        
    Returns:
        dict: Dictionary with status values as keys and pooled selectivity data as values
    """
    # Load metadata
    metadata = load_animal_metadata(data_path)
    
    # Get animals by status
    animals_by_status = get_animals_by_status(metadata, status_type, status_values)
    
    print(f"Pooling selectivity data by {status_type} for session: {session_label}")
    for status, animals in animals_by_status.items():
        print(f"  {status}: {len(animals)} animals")
    
    pooled_data = {}
    
    for status, animal_list in animals_by_status.items():
        pooled_selectivities = []
        pooled_cs_plus_responses = []
        pooled_cs_minus_responses = []
        animal_contributions = {}  # Track how many ROIs each animal contributes
        
        for animal_name in animal_list:
            try:
                # Analyze this animal's session
                results = analyze_single_animal_session(
                    data_path=data_path,
                    animal_name=animal_name,
                    session_label=session_label,
                    save_path=None,  # Don't save individual plots
                    generate_traces=False,  # Don't generate traces
                    generate_histograms=False,  # Don't generate individual histograms
                    selectivity_method=selectivity_method
                )
                
                selectivity_indices = results['selectivity_indices']
                
                # Extract valid selectivity values
                valid_selectivities = []
                valid_cs_plus = []
                valid_cs_minus = []
                
                for roi_data in selectivity_indices.values():
                    selectivity = roi_data['selectivity']
                    if not np.isnan(selectivity) and np.isfinite(selectivity):
                        valid_selectivities.append(selectivity)
                        valid_cs_plus.append(roi_data['cs_plus_response'])
                        valid_cs_minus.append(roi_data['cs_minus_response'])
                
                # Add to pooled data
                pooled_selectivities.extend(valid_selectivities)
                pooled_cs_plus_responses.extend(valid_cs_plus)
                pooled_cs_minus_responses.extend(valid_cs_minus)
                
                # Track animal contributions
                animal_contributions[animal_name] = len(valid_selectivities)
                
            except Exception as e:
                animal_contributions[animal_name] = 0
        
        pooled_data[status] = {
            'selectivities': np.array(pooled_selectivities),
            'cs_plus_responses': np.array(pooled_cs_plus_responses),
            'cs_minus_responses': np.array(pooled_cs_minus_responses),
            'animal_contributions': animal_contributions,
            'total_rois': len(pooled_selectivities),
            'total_animals': len([a for a, count in animal_contributions.items() if count > 0])
        }
    
    return pooled_data

def plot_pooled_selectivity_histograms(pooled_data, status_type='learner', session_label=None, 
                                     save_path=None, figsize=(12, 6), colors=None, alpha=0.6):
    """
    Plot overlaid histograms comparing selectivity distributions between different status groups.
    
    Args:
        pooled_data (dict): Pooled selectivity data from pool_selectivity_data_by_status()
        status_type (str): Type of status being compared
        session_label (str): Session label for title
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
        colors (dict): Colors for each status group
        alpha (float): Transparency for histogram bars
    """
    if colors is None:
        colors = {
            'learner': '#1f77b4',
            'non-learner': '#ff7f0e',
            'cheater': '#1f77b4',
            'non-cheater': '#ff7f0e',
            'male': '#1f77b4',
            'female': 'red',
            'c1_c2': '#1f77b4',
            'other': '#ff7f0e'
        }
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Define consistent bins
    bin_edges = np.arange(-1, 1.2, 0.2)  # Bins from -1 to 1 with width 0.2
    
    # Plot histogram for each status group
    for status, data in pooled_data.items():
        selectivities = data['selectivities']
        total_rois = data['total_rois']
        total_animals = data['total_animals']
        
        if len(selectivities) > 0:
            color = colors.get(status, 'gray')
            label = f"{status} (n={total_rois} ROIs, {total_animals} mice)"
            
            ax.hist(selectivities, bins=bin_edges, alpha=alpha, edgecolor='black', 
                   color=color, label=label, density=False)
    
    # Formatting
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Selectivity Index', fontsize=16)
    ax.set_ylabel('Number of ROIs', fontsize=16)
    
    # Create title
    if session_label:
        title = f'Selectivity Indices by {status_type.title()} Status - {session_label}'
    else:
        title = f'Selectivity Indices by {status_type.title()} Status'
    
    ax.set_title(title + '\n(CS+ - CS-) / (CS+ + CS-)', fontsize=18, fontweight='bold')
    ax.legend()
    
    # Add statistical summary
    stats_text = []
    for status, data in pooled_data.items():
        selectivities = data['selectivities']
        if len(selectivities) > 0:
            mean_si = np.mean(selectivities)
            mean_abs_si = np.mean(np.abs(selectivities))
            stats_text.append(f"{status}: Mean SI = {mean_si:.3f}, Mean |SI| = {mean_abs_si:.3f}")
    
    if stats_text:
        ax.text(0.02, 0.98, '\n'.join(stats_text), 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if session_label:
            filename = f'pooled_selectivity_{status_type}_{session_label}.png'
        else:
            filename = f'pooled_selectivity_{status_type}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved pooled selectivity histogram to {save_path}")
    
    plt.show()

def pool_selectivity_data_by_training_phase(data_path, status_type='learner', training_phase='early',
                                          status_values=None, selectivity_method='peak'):
    """
    Pool selectivity data across multiple sessions within a training phase.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        status_type (str): Type of status to group by ('learner', 'cheater', etc.)
        training_phase (str): 'early' (Day1, Day2) or 'late' (Day3+)
        status_values (list): List of status values to include
        selectivity_method (str): Method for selectivity calculation
        
    Returns:
        dict: Dictionary with status values as keys and pooled selectivity data as values
    """
    # Load data to get available sessions
    all_data = load_processed_data(data_path)
    
    # Determine which sessions belong to this training phase
    phase_sessions = []
    if training_phase == 'early':
        # Early training: Day1, Day2
        phase_sessions = ['Day1', 'Day2']
    elif training_phase == 'late':
        # Late training: Day3 and beyond
        phase_sessions = [f'Day{i}' for i in range(3, 11)]  # Day3 through Day10 (should cover most cases)
    
    # Find which sessions actually exist in the data
    existing_sessions = set()
    for animal_data in all_data.values():
        sessions = animal_data['sessions']
        for session_data in sessions.values():
            existing_sessions.add(session_data['label'])
    
    # Filter to only sessions that exist and match our phase
    available_phase_sessions = [s for s in phase_sessions if s in existing_sessions]
    
    if not available_phase_sessions:
        print(f"No {training_phase} training sessions found. Available sessions: {sorted(existing_sessions)}")
        return {}
    
    print(f"Pooling {training_phase} training sessions: {available_phase_sessions}")
    
    # Load metadata
    metadata = load_animal_metadata(data_path)
    
    # Get animals by status
    animals_by_status = get_animals_by_status(metadata, status_type, status_values)
    
    print(f"Pooling selectivity data by {status_type} for {training_phase} training phase")
    for status, animals in animals_by_status.items():
        print(f"  {status}: {len(animals)} animals")
    
    pooled_data = {}
    
    for status, animal_list in animals_by_status.items():
        pooled_selectivities = []
        pooled_cs_plus_responses = []
        pooled_cs_minus_responses = []
        animal_contributions = {}  # Track how many ROIs each animal contributes
        session_contributions = {}  # Track contributions by session
        
        for animal_name in animal_list:
            animal_total_rois = 0
            
            # Pool across all available sessions for this phase
            for session_label in available_phase_sessions:
                try:
                    # Analyze this animal's session
                    results = analyze_single_animal_session(
                        data_path=data_path,
                        animal_name=animal_name,
                        session_label=session_label,
                        save_path=None,  # Don't save individual plots
                        generate_traces=False,  # Don't generate traces
                        generate_histograms=False,  # Don't generate individual histograms
                        selectivity_method=selectivity_method
                    )
                    
                    selectivity_indices = results['selectivity_indices']
                    
                    # Extract valid selectivity values
                    valid_selectivities = []
                    valid_cs_plus = []
                    valid_cs_minus = []
                    
                    for roi_data in selectivity_indices.values():
                        selectivity = roi_data['selectivity']
                        if not np.isnan(selectivity) and np.isfinite(selectivity):
                            valid_selectivities.append(selectivity)
                            valid_cs_plus.append(roi_data['cs_plus_response'])
                            valid_cs_minus.append(roi_data['cs_minus_response'])
                    
                    # Add to pooled data
                    pooled_selectivities.extend(valid_selectivities)
                    pooled_cs_plus_responses.extend(valid_cs_plus)
                    pooled_cs_minus_responses.extend(valid_cs_minus)
                    
                    # Track contributions
                    animal_total_rois += len(valid_selectivities)
                    session_key = f"{animal_name}_{session_label}"
                    session_contributions[session_key] = len(valid_selectivities)
                    
                    # Track individual animal contributions
                    
                except Exception as e:
                    session_key = f"{animal_name}_{session_label}"
                    session_contributions[session_key] = 0
            
            animal_contributions[animal_name] = animal_total_rois
        
        pooled_data[status] = {
            'selectivities': np.array(pooled_selectivities),
            'cs_plus_responses': np.array(pooled_cs_plus_responses),
            'cs_minus_responses': np.array(pooled_cs_minus_responses),
            'animal_contributions': animal_contributions,
            'session_contributions': session_contributions,
            'total_rois': len(pooled_selectivities),
            'total_animals': len([a for a, count in animal_contributions.items() if count > 0]),
            'sessions_included': available_phase_sessions
        }
        
        print(f"  {status} total: {len(pooled_selectivities)} ROIs from {pooled_data[status]['total_animals']} animals across {len(available_phase_sessions)} sessions")
    
    return pooled_data

def plot_pooled_selectivity_comparison_across_phases(data_path, status_type='learner', 
                                                   status_values=None, save_path=None, 
                                                   figsize=(20, 5), selectivity_method='peak'):
    """
    Plot selectivity histograms comparing status groups across all 4 phases in a single figure.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        status_type (str): Type of status to compare ('learner', 'cheater', etc.)
        status_values (list): List of status values to compare
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
        selectivity_method (str): Method for selectivity calculation
    """
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f'Selectivity Indices by {status_type.title()} Status - All Phases', 
                 fontsize=16, fontweight='bold')
    
    phase_names = ['Pre', 'Early training', 'Late training', 'Post']
    
    for i, phase_name in enumerate(phase_names):
        ax = axes[i]
        
        try:
            if phase_name == 'Pre':
                # Pool data for Pre session
                pooled_data = pool_selectivity_data_by_status(
                    data_path=data_path,
                    status_type=status_type,
                    session_label='Pre',
                    status_values=status_values,
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Post':
                # Pool data for Post session
                pooled_data = pool_selectivity_data_by_status(
                    data_path=data_path,
                    status_type=status_type,
                    session_label='Post',
                    status_values=status_values,
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Early training':
                # Pool data across early training sessions (Day1, Day2)
                pooled_data = pool_selectivity_data_by_training_phase(
                    data_path=data_path,
                    status_type=status_type,
                    training_phase='early',
                    status_values=status_values,
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Late training':
                # Pool data across late training sessions (Day3+)
                pooled_data = pool_selectivity_data_by_training_phase(
                    data_path=data_path,
                    status_type=status_type,
                    training_phase='late',
                    status_values=status_values,
                    selectivity_method=selectivity_method
                )
            
            if not pooled_data:
                ax.text(0.5, 0.5, f'No {phase_name}\ndata found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(phase_name, fontsize=14, fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1)
                continue
            
            # Plot histograms for this phase
            colors = {
                'learner': '#1f77b4',
                'non-learner': '#ff7f0e',
                'cheater': '#1f77b4',
                'non-cheater': '#ff7f0e',
                'c1_c2': '#1f77b4',
                'other': '#ff7f0e'
            }
            
            bin_edges = np.arange(-1, 1.2, 0.2)
            
            for status, data in pooled_data.items():
                selectivities = data['selectivities']
                if len(selectivities) > 0:
                    color = colors.get(status, 'gray')
                    label = f"{status}"  # Label without ROI count for color legend
                    ax.hist(selectivities, bins=bin_edges, alpha=0.6, edgecolor='black',
                           color=color, label=label)
            
            # Add mean |SI| and ROI counts as text for each phase
            stats_text = []
            for status, data in pooled_data.items():
                if len(data['selectivities']) > 0:
                    mean_abs_si = np.mean(np.abs(data['selectivities']))
                    stats_text.append(f"{status}: mean |SI| = {mean_abs_si:.3f}")
                    stats_text.append(f"n = {data['total_rois']} ROIs")
            
            if stats_text:
                ax.text(0.02, 0.98, '\n'.join(stats_text), 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
            
            ax.set_xlim(-1, 1)
            ax.set_xlabel('Selectivity Index', fontsize=16)
            if i == 0:  # Only label y-axis for leftmost plot
                ax.set_ylabel('Number of ROIs', fontsize=16)
            ax.set_title(phase_name, fontsize=18, fontweight='bold')
            if i == 3:  # Only show color legend on the last panel (Post)
                ax.legend(fontsize=12)
            ax.grid(False)
            
        except Exception as e:
            print(f"Error processing {phase_name}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{phase_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(phase_name, fontsize=14, fontweight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f'pooled_selectivity_{status_type}_all_phases.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved pooled selectivity comparison to {save_path}")
    
    plt.show()

def pool_all_selectivity_data(data_path, session_label=None, selectivity_method='peak'):
    """
    Pool selectivity indices across ALL mice without any status grouping.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        session_label (str): Specific session to analyze (e.g., 'Pre', 'Day1', 'Post')
        selectivity_method (str): Method for selectivity calculation ('peak' or 'mean')
        
    Returns:
        dict: Dictionary containing pooled selectivity data for all animals
    """
    # Load metadata to get all animals
    metadata = load_animal_metadata(data_path)
    all_animals = list(metadata.keys())
    
    print(f"Pooling selectivity data for ALL animals (no separation) for session: {session_label}")
    print(f"  Total animals: {len(all_animals)}")
    
    pooled_selectivities = []
    pooled_cs_plus_responses = []
    pooled_cs_minus_responses = []
    animal_contributions = {}  # Track how many ROIs each animal contributes
    
    for animal_name in all_animals:
        try:
            # Analyze this animal's session
            results = analyze_single_animal_session(
                data_path=data_path,
                animal_name=animal_name,
                session_label=session_label,
                save_path=None,  # Don't save individual plots
                generate_traces=False,  # Don't generate traces
                generate_histograms=False,  # Don't generate individual histograms
                selectivity_method=selectivity_method
            )
            
            selectivity_indices = results['selectivity_indices']
            
            # Extract valid selectivity values
            valid_selectivities = []
            valid_cs_plus = []
            valid_cs_minus = []
            
            for roi_data in selectivity_indices.values():
                selectivity = roi_data['selectivity']
                if not np.isnan(selectivity) and np.isfinite(selectivity):
                    valid_selectivities.append(selectivity)
                    valid_cs_plus.append(roi_data['cs_plus_response'])
                    valid_cs_minus.append(roi_data['cs_minus_response'])
            
            # Add to pooled data
            pooled_selectivities.extend(valid_selectivities)
            pooled_cs_plus_responses.extend(valid_cs_plus)
            pooled_cs_minus_responses.extend(valid_cs_minus)
            
            # Track animal contributions
            animal_contributions[animal_name] = len(valid_selectivities)
            
        except Exception as e:
            animal_contributions[animal_name] = 0
    
    pooled_data = {
        'selectivities': np.array(pooled_selectivities),
        'cs_plus_responses': np.array(pooled_cs_plus_responses),
        'cs_minus_responses': np.array(pooled_cs_minus_responses),
        'animal_contributions': animal_contributions,
        'total_rois': len(pooled_selectivities),
        'total_animals': len([a for a, count in animal_contributions.items() if count > 0])
    }
    
    print(f"  Total pooled: {len(pooled_selectivities)} ROIs from {pooled_data['total_animals']} animals")
    
    return pooled_data

def pool_all_selectivity_data_by_training_phase(data_path, training_phase='early', selectivity_method='peak'):
    """
    Pool selectivity data across multiple sessions within a training phase for ALL animals.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        training_phase (str): 'early' (Day1, Day2) or 'late' (Day3+)
        selectivity_method (str): Method for selectivity calculation
        
    Returns:
        dict: Dictionary containing pooled selectivity data for all animals
    """
    # Load data to get available sessions
    all_data = load_processed_data(data_path)
    
    # Determine which sessions belong to this training phase
    phase_sessions = []
    if training_phase == 'early':
        # Early training: Day1, Day2
        phase_sessions = ['Day1', 'Day2']
    elif training_phase == 'late':
        # Late training: Day3 and beyond
        phase_sessions = [f'Day{i}' for i in range(3, 11)]  # Day3 through Day10 (should cover most cases)
    
    # Find which sessions actually exist in the data
    existing_sessions = set()
    for animal_data in all_data.values():
        sessions = animal_data['sessions']
        for session_data in sessions.values():
            existing_sessions.add(session_data['label'])
    
    # Filter to only sessions that exist and match our phase
    available_phase_sessions = [s for s in phase_sessions if s in existing_sessions]
    
    if not available_phase_sessions:
        print(f"No {training_phase} training sessions found. Available sessions: {sorted(existing_sessions)}")
        return {}
    
    print(f"Pooling {training_phase} training sessions for ALL animals: {available_phase_sessions}")
    
    # Load metadata to get all animals
    metadata = load_animal_metadata(data_path)
    all_animals = list(metadata.keys())
    
    print(f"Pooling selectivity data for ALL animals (no grouping) for {training_phase} training phase")
    print(f"  Total animals: {len(all_animals)}")
    
    pooled_selectivities = []
    pooled_cs_plus_responses = []
    pooled_cs_minus_responses = []
    animal_contributions = {}  # Track how many ROIs each animal contributes
    session_contributions = {}  # Track contributions by session
    
    for animal_name in all_animals:
        animal_total_rois = 0
        
        # Pool across all available sessions for this phase
        for session_label in available_phase_sessions:
            try:
                # Analyze this animal's session
                results = analyze_single_animal_session(
                    data_path=data_path,
                    animal_name=animal_name,
                    session_label=session_label,
                    save_path=None,  # Don't save individual plots
                    generate_traces=False,  # Don't generate traces
                    generate_histograms=False,  # Don't generate individual histograms
                    selectivity_method=selectivity_method
                )
                
                selectivity_indices = results['selectivity_indices']
                
                # Extract valid selectivity values
                valid_selectivities = []
                valid_cs_plus = []
                valid_cs_minus = []
                
                for roi_data in selectivity_indices.values():
                    selectivity = roi_data['selectivity']
                    if not np.isnan(selectivity) and np.isfinite(selectivity):
                        valid_selectivities.append(selectivity)
                        valid_cs_plus.append(roi_data['cs_plus_response'])
                        valid_cs_minus.append(roi_data['cs_minus_response'])
                
                # Add to pooled data
                pooled_selectivities.extend(valid_selectivities)
                pooled_cs_plus_responses.extend(valid_cs_plus)
                pooled_cs_minus_responses.extend(valid_cs_minus)
                
                # Track contributions
                animal_total_rois += len(valid_selectivities)
                session_key = f"{animal_name}_{session_label}"
                session_contributions[session_key] = len(valid_selectivities)
                
                # Track individual animal contributions
                
            except Exception as e:
                session_key = f"{animal_name}_{session_label}"
                session_contributions[session_key] = 0
        
        animal_contributions[animal_name] = animal_total_rois
    
    pooled_data = {
        'selectivities': np.array(pooled_selectivities),
        'cs_plus_responses': np.array(pooled_cs_plus_responses),
        'cs_minus_responses': np.array(pooled_cs_minus_responses),
        'animal_contributions': animal_contributions,
        'session_contributions': session_contributions,
        'total_rois': len(pooled_selectivities),
        'total_animals': len([a for a, count in animal_contributions.items() if count > 0]),
        'sessions_included': available_phase_sessions
    }
    
    print(f"  Total pooled: {len(pooled_selectivities)} ROIs from {pooled_data['total_animals']} animals across {len(available_phase_sessions)} sessions")
    
    return pooled_data

def plot_all_pooled_selectivity_histogram(pooled_data, session_label=None, save_path=None, 
                                        figsize=(10, 6), color='gray', alpha=0.7):
    """
    Plot histogram for all pooled selectivity data (no status separation).
    
    Args:
        pooled_data (dict): Pooled selectivity data from pool_all_selectivity_data()
        session_label (str): Session label for title
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
        color (str): Color for histogram bars
        alpha (float): Transparency for histogram bars
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    selectivities = pooled_data['selectivities']
    total_rois = pooled_data['total_rois']
    total_animals = pooled_data['total_animals']
    
    if len(selectivities) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
    else:
        # Define consistent bins
        bin_edges = np.arange(-1, 1.2, 0.2)  # Bins from -1 to 1 with width 0.2
        
        # Plot histogram
        label = f"All animals (n={total_rois} ROIs, {total_animals} mice)"
        ax.hist(selectivities, bins=bin_edges, alpha=alpha, edgecolor='black', 
               color=color, label=label, density=False)
    
    # Formatting
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Selectivity Index', fontsize=16)
    ax.set_ylabel('Number of ROIs', fontsize=16)
    
    # Create title
    if session_label:
        title = f'Selectivity Indices - All Animals Pooled - {session_label}'
    else:
        title = f'Selectivity Indices - All Animals Pooled'
    
    ax.set_title(title + '\n(CS+ - CS-) / (CS+ + CS-)', fontsize=18, fontweight='bold')
    
    # Add statistical summary
    if len(selectivities) > 0:
        mean_si = np.mean(selectivities)
        mean_abs_si = np.mean(np.abs(selectivities))
        median_si = np.median(selectivities)
        std_si = np.std(selectivities)
        
        stats_text = f"Mean SI = {mean_si:.3f} ± {std_si:.3f}\nMedian SI = {median_si:.3f}\nMean |SI| = {mean_abs_si:.3f}"
        
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if session_label:
            filename = f'pooled_selectivity_all_animals_{session_label}.png'
        else:
            filename = f'pooled_selectivity_all_animals.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved all-animals pooled selectivity histogram to {save_path}")
    
    plt.show()

def plot_all_pooled_selectivity_comparison_across_phases(data_path, save_path=None, 
                                                       figsize=(20, 5), selectivity_method='peak'):
    """
    Plot selectivity histograms for all animals pooled together across all 4 phases in a single figure.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str): Path to save the plot
        figsize (tuple): Figure size
        selectivity_method (str): Method for selectivity calculation
    """
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle('Selectivity Indices - All Animals Pooled - All Phases', 
                 fontsize=16, fontweight='bold')
    
    phase_names = ['Pre', 'Early training', 'Late training', 'Post']
    
    for i, phase_name in enumerate(phase_names):
        ax = axes[i]
        
        try:
            if phase_name == 'Pre':
                # Pool data for Pre session
                pooled_data = pool_all_selectivity_data(
                    data_path=data_path,
                    session_label='Pre',
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Post':
                # Pool data for Post session
                pooled_data = pool_all_selectivity_data(
                    data_path=data_path,
                    session_label='Post',
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Early training':
                # Pool data across early training sessions (Day1, Day2)
                pooled_data = pool_all_selectivity_data_by_training_phase(
                    data_path=data_path,
                    training_phase='early',
                    selectivity_method=selectivity_method
                )
            elif phase_name == 'Late training':
                # Pool data across late training sessions (Day3+)
                pooled_data = pool_all_selectivity_data_by_training_phase(
                    data_path=data_path,
                    training_phase='late',
                    selectivity_method=selectivity_method
                )
            
            if not pooled_data or len(pooled_data.get('selectivities', [])) == 0:
                ax.text(0.5, 0.5, f'No {phase_name}\ndata found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(phase_name, fontsize=14, fontweight='bold')
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1)
                continue
            
            # Plot histogram for this phase
            selectivities = pooled_data['selectivities']
            bin_edges = np.arange(-1, 1.2, 0.2)
            
            label = f"All"  # Label without ROI count for color legend
            ax.hist(selectivities, bins=bin_edges, alpha=0.7, edgecolor='black',
                   color='gray', label=label)
            
            # Add mean |SI| and ROI count as text for each phase
            if len(selectivities) > 0:
                mean_abs_si = np.mean(np.abs(selectivities))
                stats_text = f"mean |SI| = {mean_abs_si:.3f}\nn = {pooled_data['total_rois']} ROIs"
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
            
            ax.set_xlim(-1, 1)
            ax.set_xlabel('Selectivity Index', fontsize=16)
            if i == 0:  # Only label y-axis for leftmost plot
                ax.set_ylabel('Number of ROIs', fontsize=16)
            ax.set_title(phase_name, fontsize=18, fontweight='bold')
            ax.grid(False)
            
        except Exception as e:
            print(f"Error processing {phase_name}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{phase_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(phase_name, fontsize=14, fontweight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f'pooled_selectivity_all_animals_all_phases.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Saved all-animals pooled selectivity comparison to {save_path}")
    
    plt.show()

def print_pooled_statistics(pooled_data, status_type='learner'):
    """
    Print detailed statistics for pooled selectivity data.
    
    Args:
        pooled_data (dict): Pooled selectivity data
        status_type (str): Type of status being compared
    """
    print(f"\nPooled Selectivity Statistics by {status_type.title()} Status")
    print("=" * 60)
    
    for status, data in pooled_data.items():
        selectivities = data['selectivities']
        cs_plus = data['cs_plus_responses']
        cs_minus = data['cs_minus_responses']
        contributions = data['animal_contributions']
        
        print(f"\n{status.upper()} GROUP:")
        print(f"  Total ROIs: {len(selectivities)}")
        print(f"  Total Animals: {data['total_animals']}")
        print(f"  Animal contributions: {dict(contributions)}")
        
        if len(selectivities) > 0:
            print(f"  Selectivity - Mean: {np.mean(selectivities):.4f} ± {np.std(selectivities):.4f}")
            print(f"  Selectivity - Median: {np.median(selectivities):.4f}")
            print(f"  Selectivity - Range: [{np.min(selectivities):.4f}, {np.max(selectivities):.4f}]")
            print(f"  Mean |Selectivity|: {np.mean(np.abs(selectivities)):.4f}")
            print(f"  CS+ Response - Mean: {np.mean(cs_plus):.4f} ± {np.std(cs_plus):.4f}")
            print(f"  CS- Response - Mean: {np.mean(cs_minus):.4f} ± {np.std(cs_minus):.4f}")

def print_all_pooled_statistics(pooled_data):
    """
    Print detailed statistics for all-animals pooled selectivity data.
    
    Args:
        pooled_data (dict): Pooled selectivity data for all animals
    """
    print(f"\nPooled Selectivity Statistics - All Animals")
    print("=" * 60)
    
    selectivities = pooled_data['selectivities']
    cs_plus = pooled_data['cs_plus_responses']
    cs_minus = pooled_data['cs_minus_responses']
    contributions = pooled_data['animal_contributions']
    
    print(f"\nALL ANIMALS POOLED:")
    print(f"  Total ROIs: {len(selectivities)}")
    print(f"  Total Animals: {pooled_data['total_animals']}")
    print(f"  Animal contributions: {dict(contributions)}")
    
    if len(selectivities) > 0:
        print(f"  Selectivity - Mean: {np.mean(selectivities):.4f} ± {np.std(selectivities):.4f}")
        print(f"  Selectivity - Median: {np.median(selectivities):.4f}")
        print(f"  Selectivity - Range: [{np.min(selectivities):.4f}, {np.max(selectivities):.4f}]")
        print(f"  Mean |Selectivity|: {np.mean(np.abs(selectivities)):.4f}")
        print(f"  CS+ Response - Mean: {np.mean(cs_plus):.4f} ± {np.std(cs_plus):.4f}")
        print(f"  CS- Response - Mean: {np.mean(cs_minus):.4f} ± {np.std(cs_minus):.4f}")

def analyze_pooled_cs_responses_by_status(data_path, status_type='learner', session_label=None,
                                        status_values=None, save_path=None, 
                                        selectivity_method='peak', generate_across_phases=True):
    """
    Main function to analyze CS responses pooled by animal status.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        status_type (str): Type of status to group by ('learner', 'cheater', 'sex', etc.)
        session_label (str): Specific session to analyze (None for across phases analysis)
        status_values (list): List of status values to include
        save_path (str): Path to save plots
        selectivity_method (str): Method for selectivity calculation
        generate_across_phases (bool): Whether to generate across-phases comparison
    """
    # Load and display metadata
    metadata = load_animal_metadata(data_path)
    animals_by_status = get_animals_by_status(metadata, status_type, status_values)
    
    print(f"\nAvailable animals by {status_type} status:")
    for status, animals in animals_by_status.items():
        print(f"  {status}: {len(animals)} animals")
    
    if session_label:
        # Analyze specific session
        print(f"\nAnalyzing session: {session_label}")
        pooled_data = pool_selectivity_data_by_status(
            data_path=data_path,
            status_type=status_type,
            session_label=session_label,
            status_values=status_values,
            selectivity_method=selectivity_method
        )
        
        # Print statistics
        print_pooled_statistics(pooled_data, status_type)
        
        # Plot histogram
        plot_pooled_selectivity_histograms(
            pooled_data=pooled_data,
            status_type=status_type,
            session_label=session_label,
            save_path=save_path
        )
    
    if generate_across_phases:
        # Analyze across all phases
        print(f"\nGenerating across-phases comparison...")
        plot_pooled_selectivity_comparison_across_phases(
            data_path=data_path,
            status_type=status_type,
            status_values=status_values,
            save_path=save_path,
            selectivity_method=selectivity_method
        )

def analyze_all_pooled_cs_responses(data_path, session_label=None, save_path=None, 
                                  selectivity_method='peak', generate_across_phases=True):
    """
    Main function to analyze CS responses pooled across ALL animals (no status separation).
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        session_label (str): Specific session to analyze (None for across phases analysis)
        save_path (str): Path to save plots
        selectivity_method (str): Method for selectivity calculation
        generate_across_phases (bool): Whether to generate across-phases comparison
    """
    print(f"Analyzing CS responses pooled across ALL animals (no separation)...")
    
    # Load and display metadata
    metadata = load_animal_metadata(data_path)
    all_animals = list(metadata.keys())
    
    print(f"\nAll animals to be pooled: {len(all_animals)} animals")
    
    if session_label:
        # Analyze specific session
        print(f"\nAnalyzing session: {session_label}")
        pooled_data = pool_all_selectivity_data(
            data_path=data_path,
            session_label=session_label,
            selectivity_method=selectivity_method
        )
        
        # Print statistics
        print_all_pooled_statistics(pooled_data)
        
        # Plot histogram
        plot_all_pooled_selectivity_histogram(
            pooled_data=pooled_data,
            session_label=session_label,
            save_path=save_path
        )
    
    if generate_across_phases:
        # Analyze across all phases
        print(f"\nGenerating across-phases comparison...")
        plot_all_pooled_selectivity_comparison_across_phases(
            data_path=data_path,
            save_path=save_path,
            selectivity_method=selectivity_method
        )

if __name__ == "__main__":
    # Set paths
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'selectivity_histograms/pooled')
    
    # Example usage
    print("=" * 60)
    print("POOLED SELECTIVITY HISTOGRAM ANALYSIS")
    print("=" * 60)
    
    # Choose what to analyze
    print("\nChoose how to pool:")
    print("1. Learner vs Non-learner")
    print("2. Cheater vs Non-cheater")
    print("3. Male vs Female")
    print("4. Intrinsic imaging (c1_c2 vs other)")
    print("5. Custom status analysis")
    print("6. All animals pooled (no separation)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            if choice == '1':
                status_type = 'learner'
                status_values = ['learner', 'non-learner']
                break
            elif choice == '2':
                status_type = 'cheater'
                status_values = ['cheater', 'non-cheater']
                break
            elif choice == '3':
                status_type = 'sex'
                status_values = ['male', 'female']
                break
            elif choice == '4':
                status_type = 'intrinsic_imaging_result'
                status_values = None  # Will be handled automatically by get_animals_by_status
                break
            elif choice == '5':
                status_type = input("Enter status type (learner/cheater/sex/intrinsic_imaging_result): ").strip()
                status_values = input("Enter status values (comma-separated): ").strip().split(',')
                status_values = [s.strip() for s in status_values]
                break
            elif choice == '6':
                # Special case for all animals pooled
                status_type = None
                status_values = None
                break
            else:
                print("Please enter 1, 2, 3, 4, 5, or 6")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
    
    # Always run across all phases
    print("\nRunning analysis across all learning phases...")
    session_label = None
    generate_across_phases = True
    
    # Run the analysis
    if status_type is None:
        # All animals pooled analysis
        analyze_all_pooled_cs_responses(
            data_path=data_path,
            session_label=session_label,
            save_path=save_path,
            selectivity_method='peak',  # You can change this to 'mean' if preferred
            generate_across_phases=generate_across_phases
        )
    else:
        # Status-based comparison analysis
        analyze_pooled_cs_responses_by_status(
            data_path=data_path,
            status_type=status_type,
            session_label=session_label,
            status_values=status_values,
            save_path=save_path,
            selectivity_method='peak',  # You can change this to 'mean' if preferred
            generate_across_phases=generate_across_phases
        )



