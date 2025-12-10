"""
Plots histograms of peak response amplitudes for all 4 learning phases in a 2x4 subplot layout.
Top row: CS+ histograms, Bottom row: CS- histograms

User must set data and save path for plots in the main function

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))

# Import necessary functions
from data_loader import load_processed_data
from learning_phase_mapping import day_label_to_learning_phase  # noqa: E402
from set_paths import processed_data_path, save_figures_path  # noqa: E402


def extract_peak_amplitudes(dff, stim_frames, frame_rate=30):
    """
    Extract peak amplitudes for baseline and stimulus periods.
    
    For each trial:
    - Baseline: 1s before stimulus onset (30 frames)
    - Stimulus: From stimulus onset to 3s (90 frames)
    - Calculates: max amplitude = stim_peak - baseline_peak
    
    Args:
        dff (np.ndarray): dF/F data (ROIs x time)
        stim_frames (list): List of (start_frame, stop_frame) tuples for each trial
        frame_rate (int): Frame rate in Hz (default: 30)
        
    Returns:
        dict: Dictionary with peak amplitudes for each trial and ROI
            Keys: trial indices
            Values: dict with 'baseline_peaks', 'stim_peaks', 'max_amplitudes' (arrays of length n_rois)
    """
    baseline_duration = 30  # 1 second before stim onset at 30 Hz
    stim_duration = 90  # 3 seconds after stim onset at 30 Hz
    
    trial_peak_data = {}
    
    for trial_idx, (start_frame, stop_frame) in enumerate(stim_frames):
        # Define baseline window: 1s before stimulus onset
        baseline_start = max(0, start_frame - baseline_duration)
        baseline_end = start_frame
        
        # Define stimulus window: from onset to 3s after
        stim_start = start_frame
        stim_end = min(dff.shape[1], start_frame + stim_duration)
        
        # Extract data for all ROIs
        baseline_data = dff[:, baseline_start:baseline_end]  # Shape: (n_rois, baseline_frames)
        stim_data = dff[:, stim_start:stim_end]  # Shape: (n_rois, stim_frames)
        
        # Skip trials with insufficient data
        if baseline_data.shape[1] < baseline_duration or stim_data.shape[1] < stim_duration:
            continue
        
        # Find peak amplitudes for each ROI
        baseline_peaks = np.max(baseline_data, axis=1)  # Peak in baseline period
        stim_peaks = np.max(stim_data, axis=1)  # Peak in stimulus period
        
        # Calculate max amplitude: stim peak - baseline peak
        max_amplitudes = stim_peaks - baseline_peaks
        
        trial_peak_data[trial_idx] = {
            'baseline_peaks': baseline_peaks,
            'stim_peaks': stim_peaks,
            'max_amplitudes': max_amplitudes
        }
    
    return trial_peak_data


def analyze_animal_session_peaks(all_data, animal_name, session_label):
    """
    Analyze peak responses for a single animal's session.
    
    Args:
        all_data (dict): Pre-loaded processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Session label (e.g., 'Pre', 'Day1', 'Post')
        
    Returns:
        dict: Dictionary with CS+ and CS- peak amplitudes across all ROIs and trials
    """
    
    if animal_name not in all_data:
        raise ValueError(f"Animal {animal_name} not found in data")
    
    animal_data = all_data[animal_name]
    
    # Find session with matching label
    session_idx = None
    for idx, session_data in animal_data['sessions'].items():
        if session_data['label'] == session_label:
            session_idx = idx
            break
    
    if session_idx is None:
        raise ValueError(f"Session {session_label} not found for animal {animal_name}")
    
    session_data = animal_data['sessions'][session_idx]
    
    # Extract necessary data
    dff = session_data['dff']  # Shape: (n_rois, n_frames)
    
    # Get stimulus frame information
    if 'cs_plus_frames' not in session_data or 'cs_minus_frames' not in session_data:
        raise ValueError(f"No stimulus frame data found for {animal_name} session {session_label}")
    
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    
    # Extract peak amplitudes for CS+ and CS- trials
    cs_plus_peaks = extract_peak_amplitudes(dff, cs_plus_frames)
    cs_minus_peaks = extract_peak_amplitudes(dff, cs_minus_frames)
    
    # Collect all max amplitudes across trials and ROIs
    cs_plus_max_amps = []
    for trial_data in cs_plus_peaks.values():
        cs_plus_max_amps.extend(trial_data['max_amplitudes'])
    
    cs_minus_max_amps = []
    for trial_data in cs_minus_peaks.values():
        cs_minus_max_amps.extend(trial_data['max_amplitudes'])
    
    return {
        'cs_plus_amplitudes': np.array(cs_plus_max_amps),
        'cs_minus_amplitudes': np.array(cs_minus_max_amps),
        'n_rois': dff.shape[0],
        'n_cs_plus_trials': len(cs_plus_frames),
        'n_cs_minus_trials': len(cs_minus_frames)
    }


def analyze_trial_averaged_peaks_single_animal(all_data, animal_name, session_label):
    """
    Analyze trial-averaged peak responses for a single animal's session.
    For each ROI, average across all CS+ trials and all CS- trials separately.
    
    Args:
        all_data (dict): Pre-loaded processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Session label (e.g., 'Pre', 'Day1', 'Post')
        
    Returns:
        dict: Dictionary with trial-averaged peak amplitudes for each ROI
    """
    
    if animal_name not in all_data:
        raise ValueError(f"Animal {animal_name} not found in data")
    
    animal_data = all_data[animal_name]
    
    # Find session with matching label
    session_idx = None
    for idx, session_data in animal_data['sessions'].items():
        if session_data['label'] == session_label:
            session_idx = idx
            break
    
    if session_idx is None:
        raise ValueError(f"Session {session_label} not found for animal {animal_name}")
    
    session_data = animal_data['sessions'][session_idx]
    
    # Extract necessary data
    dff = session_data['dff']  # Shape: (n_rois, n_frames)
    n_rois = dff.shape[0]
    
    # Get stimulus frame information
    if 'cs_plus_frames' not in session_data or 'cs_minus_frames' not in session_data:
        raise ValueError(f"No stimulus frame data found for {animal_name} session {session_label}")
    
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    
    # Extract peak amplitudes for CS+ and CS- trials
    cs_plus_peaks = extract_peak_amplitudes(dff, cs_plus_frames)
    cs_minus_peaks = extract_peak_amplitudes(dff, cs_minus_frames)
    
    # Calculate trial-averaged amplitudes for each ROI
    cs_plus_roi_averages = np.zeros(n_rois)
    cs_minus_roi_averages = np.zeros(n_rois)
    
    # For each ROI, average across all trials
    for roi_idx in range(n_rois):
        # Collect amplitudes for this ROI across all CS+ trials
        cs_plus_roi_amplitudes = []
        for trial_data in cs_plus_peaks.values():
            cs_plus_roi_amplitudes.append(trial_data['max_amplitudes'][roi_idx])
        
        # Collect amplitudes for this ROI across all CS- trials
        cs_minus_roi_amplitudes = []
        for trial_data in cs_minus_peaks.values():
            cs_minus_roi_amplitudes.append(trial_data['max_amplitudes'][roi_idx])
        
        # Calculate mean for this ROI
        cs_plus_roi_averages[roi_idx] = np.mean(cs_plus_roi_amplitudes) if cs_plus_roi_amplitudes else 0
        cs_minus_roi_averages[roi_idx] = np.mean(cs_minus_roi_amplitudes) if cs_minus_roi_amplitudes else 0
    
    return {
        'cs_plus_roi_averages': cs_plus_roi_averages,
        'cs_minus_roi_averages': cs_minus_roi_averages,
        'n_rois': n_rois,
        'n_cs_plus_trials': len(cs_plus_frames),
        'n_cs_minus_trials': len(cs_minus_frames),
        'animal_name': animal_name,
        'session_label': session_label
    }


def pool_peak_data_by_learning_phase(all_data, learning_phase):
    """
    Pool peak response data across all animals for a specific learning phase.
    
    Args:
        all_data (dict): Pre-loaded processed data
        learning_phase (str): Learning phase ('Pre', 'Early Learning', 'Late Learning', 'Post')
        
    Returns:
        dict: Pooled peak amplitudes for CS+ and CS- across all animals
    """
    
    pooled_cs_plus = []
    pooled_cs_minus = []
    animal_contributions = {}
    
    # Iterate through all animals
    for animal_name, animal_data in all_data.items():
        # Find sessions that match this learning phase
        for session_idx, session_data in animal_data['sessions'].items():
            session_label = session_data['label']
            session_phase = day_label_to_learning_phase(session_label)
            
            if session_phase == learning_phase:
                try:
                    # Analyze this session
                    results = analyze_animal_session_peaks(
                        all_data=all_data,
                        animal_name=animal_name,
                        session_label=session_label
                    )
                    
                    # Add to pooled data
                    pooled_cs_plus.extend(results['cs_plus_amplitudes'])
                    pooled_cs_minus.extend(results['cs_minus_amplitudes'])
                    
                    # Track contributions
                    key = f"{animal_name}_{session_label}"
                    animal_contributions[key] = {
                        'n_cs_plus': len(results['cs_plus_amplitudes']),
                        'n_cs_minus': len(results['cs_minus_amplitudes'])
                    }
                    
                    print(f"  {animal_name} {session_label}: Processed")
                    
                except Exception as e:
                    print(f"  {animal_name} {session_label}: Error - {e}")
    
    return {
        'cs_plus_amplitudes': np.array(pooled_cs_plus),
        'cs_minus_amplitudes': np.array(pooled_cs_minus),
        'animal_contributions': animal_contributions,
        'n_datapoints_cs_plus': len(pooled_cs_plus),
        'n_datapoints_cs_minus': len(pooled_cs_minus)
    }


def plot_peak_amplitude_histograms(data_path, save_path=None, figsize=(20, 10)):
    """
    Plot histograms of peak amplitudes for all 4 learning phases in a 2x4 subplot layout.
    Top row: CS+ histograms, Bottom row: CS- histograms
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Load data once
    print(f"Loading data from: {data_path}")
    all_data = load_processed_data(data_path)
    
    # Define learning phases
    learning_phases = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    
    # Create figure with 8 subplots (2x4)
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharey=True)
    fig.suptitle('Peak Response Amplitudes Across Learning Phases\n(Stim Peak - Baseline Peak)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Process each learning phase
    for idx, phase in enumerate(learning_phases):
        # Get axes for CS+ (top row) and CS- (bottom row)
        ax_cs_plus = axes[0, idx]  # Top row
        ax_cs_minus = axes[1, idx]  # Bottom row
        
        print(f"\nProcessing {phase}...")
        try:
            # Pool data for this phase
            pooled_data = pool_peak_data_by_learning_phase(all_data, phase)
            
            cs_plus_amps = pooled_data['cs_plus_amplitudes']
            cs_minus_amps = pooled_data['cs_minus_amplitudes']
            
            # Define bins for histogram
            bin_edges = np.arange(0, 3.2, 0.1)  # From 0 to 3.2 with 0.1 width
            
            # Plot CS+ histogram (top row)
            if len(cs_plus_amps) == 0:
                ax_cs_plus.text(0.5, 0.5, f'No CS+ data\navailable', 
                               ha='center', va='center', transform=ax_cs_plus.transAxes, 
                               fontsize=14, fontweight='bold')
                ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
                ax_cs_plus.set_xlim(0, 3)
            else:
                ax_cs_plus.hist(cs_plus_amps, bins=bin_edges, alpha=0.6, 
                               color='#d62728', edgecolor='black', label=f'CS+ (n={len(cs_plus_amps)})')
                ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
                ax_cs_plus.set_xlim(0, 3)
                ax_cs_plus.grid(True, alpha=0.3, linestyle='--')
                
                # Add CS+ statistics
                mean_cs_plus = np.mean(cs_plus_amps)
                median_cs_plus = np.median(cs_plus_amps)
                stats_text = [f'Mean: {mean_cs_plus:.3f}', f'Median: {median_cs_plus:.3f}']
                ax_cs_plus.text(0.98, 0.97, '\n'.join(stats_text),
                               transform=ax_cs_plus.transAxes, verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9)
            
            # Plot CS- histogram (bottom row)
            if len(cs_minus_amps) == 0:
                ax_cs_minus.text(0.5, 0.5, f'No CS- data\navailable', 
                                ha='center', va='center', transform=ax_cs_minus.transAxes, 
                                fontsize=14, fontweight='bold')
                ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
                ax_cs_minus.set_xlim(0, 3)
            else:
                ax_cs_minus.hist(cs_minus_amps, bins=bin_edges, alpha=0.6, 
                                color='#1f77b4', edgecolor='black', label=f'CS- (n={len(cs_minus_amps)})')
                ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
                ax_cs_minus.set_xlim(0, 3)
                ax_cs_minus.grid(True, alpha=0.3, linestyle='--')
                
                # Add CS- statistics
                mean_cs_minus = np.mean(cs_minus_amps)
                median_cs_minus = np.median(cs_minus_amps)
                stats_text = [f'Mean: {mean_cs_minus:.3f}', f'Median: {median_cs_minus:.3f}']
                ax_cs_minus.text(0.98, 0.97, '\n'.join(stats_text),
                                transform=ax_cs_minus.transAxes, verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=9)
            
            # Set common labels
            ax_cs_plus.set_xlabel('Max Amplitude (ΔF/F)', fontsize=14)
            ax_cs_minus.set_xlabel('Max Amplitude (ΔF/F)', fontsize=14)
            if idx == 0:
                ax_cs_plus.set_ylabel('Count', fontsize=14)
                ax_cs_minus.set_ylabel('Count', fontsize=14)
            
            print(f"  {phase}: Processed")
            
        except Exception as e:
            print(f"  Error processing {phase}: {e}")
            # Handle errors for both subplots
            ax_cs_plus.text(0.5, 0.5, f'Error loading\n{phase}', 
                           ha='center', va='center', transform=ax_cs_plus.transAxes, 
                           fontsize=14, fontweight='bold')
            ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
            ax_cs_plus.set_xlim(0, 0.7)
            
            ax_cs_minus.text(0.5, 0.5, f'Error loading\n{phase}', 
                            ha='center', va='center', transform=ax_cs_minus.transAxes, 
                            fontsize=14, fontweight='bold')
            ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
            ax_cs_minus.set_xlim(0, 0.7)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = 'peak_amplitude_histograms_all_phases.png'
        full_path = os.path.join(sig_test_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to: {full_path}")
    
    plt.show()


def pool_trial_averaged_peaks_by_learning_phase(all_data, learning_phase):
    """
    Pool trial-averaged peak response data across all animals for a specific learning phase.
    For each animal, calculate trial-averaged amplitudes for each ROI, then pool across all animals.
    
    Args:
        all_data (dict): Pre-loaded processed data
        learning_phase (str): Learning phase ('Pre', 'Early Learning', 'Late Learning', 'Post')
        
    Returns:
        dict: Pooled trial-averaged peak amplitudes for CS+ and CS- across all animals
    """
    
    pooled_cs_plus_roi_averages = []
    pooled_cs_minus_roi_averages = []
    animal_contributions = {}
    
    # Iterate through all animals
    for animal_name, animal_data in all_data.items():
        # Find sessions that match this learning phase
        for session_idx, session_data in animal_data['sessions'].items():
            session_label = session_data['label']
            session_phase = day_label_to_learning_phase(session_label)
            
            if session_phase == learning_phase:
                try:
                    # Analyze trial-averaged data for this session
                    results = analyze_trial_averaged_peaks_single_animal(
                        all_data=all_data,
                        animal_name=animal_name,
                        session_label=session_label
                    )
                    
                    # Add ROI averages to pooled data
                    pooled_cs_plus_roi_averages.extend(results['cs_plus_roi_averages'])
                    pooled_cs_minus_roi_averages.extend(results['cs_minus_roi_averages'])
                    
                    # Track contributions
                    key = f"{animal_name}_{session_label}"
                    animal_contributions[key] = {
                        'n_rois': results['n_rois'],
                        'n_cs_plus_trials': results['n_cs_plus_trials'],
                        'n_cs_minus_trials': results['n_cs_minus_trials']
                    }
                    
                    print(f"  {animal_name} {session_label}: Processed")
                    
                except Exception as e:
                    print(f"  {animal_name} {session_label}: Error - {e}")
    
    return {
        'cs_plus_roi_averages': np.array(pooled_cs_plus_roi_averages),
        'cs_minus_roi_averages': np.array(pooled_cs_minus_roi_averages),
        'animal_contributions': animal_contributions,
        'n_rois_cs_plus': len(pooled_cs_plus_roi_averages),
        'n_rois_cs_minus': len(pooled_cs_minus_roi_averages)
    }




def plot_trial_averaged_peak_histograms_pooled(data_path, save_path=None, figsize=(20, 10)):
    """
    Plot histograms of trial-averaged peak amplitudes for all 4 learning phases in a 2x4 subplot layout.
    Top row: CS+ histograms, Bottom row: CS- histograms
    Each ROI is trial-averaged within each animal, then pooled across all animals.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        save_path (str): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    # Load data once
    print(f"Loading data from: {data_path}")
    all_data = load_processed_data(data_path)
    
    # Define learning phases
    learning_phases = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    
    # Create figure with 8 subplots (2x4)
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharey=True)
    fig.suptitle('Trial-Averaged Peak Response Amplitudes Across Learning Phases\n(ROI averages pooled across all animals)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Process each learning phase
    for idx, phase in enumerate(learning_phases):
        # Get axes for CS+ (top row) and CS- (bottom row)
        ax_cs_plus = axes[0, idx]  # Top row
        ax_cs_minus = axes[1, idx]  # Bottom row
        
        print(f"\nProcessing {phase}...")
        try:
            # Pool trial-averaged data for this phase
            pooled_data = pool_trial_averaged_peaks_by_learning_phase(all_data, phase)
            
            cs_plus_roi_averages = pooled_data['cs_plus_roi_averages']
            cs_minus_roi_averages = pooled_data['cs_minus_roi_averages']
            
            # Define bins for histogram
            bin_edges = np.arange(0, 0.7, 0.025)  # From 0 to 0.7 with 0.025 width
            
            # Plot CS+ histogram (top row)
            if len(cs_plus_roi_averages) == 0:
                ax_cs_plus.text(0.5, 0.5, f'No CS+ data\navailable', 
                               ha='center', va='center', transform=ax_cs_plus.transAxes, 
                               fontsize=14, fontweight='bold')
                ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
                ax_cs_plus.set_xlim(0, 0.7)
            else:
                ax_cs_plus.hist(cs_plus_roi_averages, bins=bin_edges, alpha=0.6, 
                               color='#d62728', edgecolor='black', label=f'CS+ (n={len(cs_plus_roi_averages)} ROIs)')
                ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
                ax_cs_plus.set_xlim(0, 0.7)
                ax_cs_plus.grid(True, alpha=0.3, linestyle='--')
                
                # Add CS+ statistics
                mean_cs_plus = np.mean(cs_plus_roi_averages)
                median_cs_plus = np.median(cs_plus_roi_averages)
                stats_text = [f'Mean: {mean_cs_plus:.3f}', f'Median: {median_cs_plus:.3f}']
                ax_cs_plus.text(0.98, 0.97, '\n'.join(stats_text),
                               transform=ax_cs_plus.transAxes, verticalalignment='top',
                               horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9)
            
            # Plot CS- histogram (bottom row)
            if len(cs_minus_roi_averages) == 0:
                ax_cs_minus.text(0.5, 0.5, f'No CS- data\navailable', 
                                ha='center', va='center', transform=ax_cs_minus.transAxes, 
                                fontsize=14, fontweight='bold')
                ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
                ax_cs_minus.set_xlim(0, 0.7)
            else:
                ax_cs_minus.hist(cs_minus_roi_averages, bins=bin_edges, alpha=0.6, 
                                color='#1f77b4', edgecolor='black', label=f'CS- (n={len(cs_minus_roi_averages)} ROIs)')
                ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
                ax_cs_minus.set_xlim(0, 0.7)
                ax_cs_minus.grid(True, alpha=0.3, linestyle='--')
                
                # Add CS- statistics
                mean_cs_minus = np.mean(cs_minus_roi_averages)
                median_cs_minus = np.median(cs_minus_roi_averages)
                stats_text = [f'Mean: {mean_cs_minus:.3f}', f'Median: {median_cs_minus:.3f}']
                ax_cs_minus.text(0.98, 0.97, '\n'.join(stats_text),
                                transform=ax_cs_minus.transAxes, verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=9)
            
            # Set common labels
            ax_cs_plus.set_xlabel('Mean Amplitude (ΔF/F)', fontsize=14)
            ax_cs_minus.set_xlabel('Mean Amplitude (ΔF/F)', fontsize=14)
            if idx == 0:
                ax_cs_plus.set_ylabel('Number of ROIs', fontsize=14)
                ax_cs_minus.set_ylabel('Number of ROIs', fontsize=14)
            
            print(f"  {phase}: Processed")
            
        except Exception as e:
            print(f"  Error processing {phase}: {e}")
            # Handle errors for both subplots
            ax_cs_plus.text(0.5, 0.5, f'Error loading\n{phase}', 
                           ha='center', va='center', transform=ax_cs_plus.transAxes, 
                           fontsize=14, fontweight='bold')
            ax_cs_plus.set_title(f'{phase} - CS+', fontsize=16, fontweight='bold')
            ax_cs_plus.set_xlim(0, 0.7)
            
            ax_cs_minus.text(0.5, 0.5, f'Error loading\n{phase}', 
                            ha='center', va='center', transform=ax_cs_minus.transAxes, 
                            fontsize=14, fontweight='bold')
            ax_cs_minus.set_title(f'{phase} - CS-', fontsize=16, fontweight='bold')
            ax_cs_minus.set_xlim(0, 0.7)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        sig_test_path = os.path.join(save_path, 'sig_test')
        os.makedirs(sig_test_path, exist_ok=True)
        filename = 'trial_averaged_peak_amplitude_histograms_all_phases.png'
        full_path = os.path.join(sig_test_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to: {full_path}")
    
    plt.show()


if __name__ == "__main__":
    # Get paths from centralized config
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'peak_amplitude_histogram')
    
    print("=" * 80)
    print("PEAK RESPONSE AMPLITUDE ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing peak amplitudes (stim peak - baseline peak) for CS+ and CS- trials")
    print("Baseline: 1s before stimulus onset")
    print("Stimulus: From onset to 3s after")
    print("Pooling across all animals within each learning phase")
    print("=" * 80)
    
    # Generate histograms for all learning phases
    plot_peak_amplitude_histograms(data_path, save_path)
    
    print("\n" + "=" * 80)
    print("TRIAL-AVERAGED PEAK RESPONSE ANALYSIS (POOLED)")
    print("=" * 80)
    print("\nAnalyzing trial-averaged peak amplitudes pooled across all animals")
    print("For each ROI, averaging across all CS+ and CS- trials separately within each animal")
    print("Then pooling ROI averages across all animals within each learning phase")
    print("=" * 80)
    
    # Generate pooled trial-averaged histograms for all learning phases
    plot_trial_averaged_peak_histograms_pooled(data_path, save_path)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
