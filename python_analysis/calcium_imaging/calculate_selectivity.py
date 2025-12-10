"""
Calculate selectivity indices for CS+ and CS- from calcium imaging data.

selectivity index is calculated as (CSplus - CSminus) / (CSplus + CSminus)

Optional clipping of negative df/f values to 0. 

This module provides core functions for:

"""

import numpy as np
from data_loader import load_processed_data
from extract_trial_dff import extract_trial_dff_baseline_corrected, extract_trial_dff


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


def calculate_selectivity_index(cs_plus_data, cs_minus_data, response_window_start=0, response_window_end=2, 
                               cs_plus_frames=None, cs_minus_frames=None, frame_rate=30, method='peak',
                               set_negative_to_zero = True):
    """
    Calculate selectivity index for each ROI using either peak amplitude or mean fluorescence from baseline-corrected data.
    The input data should already be baseline-corrected from extract_trial_dff_baseline_corrected.
    
    Args:
        cs_plus_data (dict): CS plus data for each ROI (baseline-corrected)
        cs_minus_data (dict): CS minus data for each ROI (baseline-corrected)
        response_window_start (float): Start of response window in seconds (relative to stimulus onset)
        response_window_end (float): End of response window in seconds (relative to stimulus onset)
        cs_plus_frames (list): List of (start_frame, stop_frame) tuples for CS+ trials
        cs_minus_frames (list): List of (start_frame, stop_frame) tuples for CS- trials
        frame_rate (int): Frame rate in Hz
        method (str): Method for calculating response values. Options:
            - 'peak': Use peak amplitude (maximum value) in the response window (default)
            - 'mean': Use mean fluorescence during the stimulus window
        set_negative_to_zero (bool): If True, set negative response values to 0 before calculating selectivity (default: True)
        
    Returns:
        dict: Selectivity index for each ROI
    """
    selectivity_indices = {}
    
    # Calculate stimulus duration from the first trial (assuming all trials have same duration)
    if cs_plus_frames is not None and len(cs_plus_frames) > 0:
        stimulus_duration = cs_plus_frames[0][1] - cs_plus_frames[0][0]  # stop_frame - start_frame
    elif cs_minus_frames is not None and len(cs_minus_frames) > 0:
        stimulus_duration = cs_minus_frames[0][1] - cs_minus_frames[0][0]  # stop_frame - start_frame
    else:
        # Fallback: assume 2-second stimulus (60 frames at 30 Hz)
        stimulus_duration = 60
        # Warning: Could not determine stimulus duration from frames, using default 2 seconds
    
    # Calculate pre_frames based on stimulus duration (go backwards from stimulus onset)
    pre_frames = stimulus_duration
    
    for roi_idx in cs_plus_data.keys():
        # Get mean traces
        cs_plus_trace = cs_plus_data[roi_idx]['mean_trace']
        cs_minus_trace = cs_minus_data[roi_idx]['mean_trace']
        
        # Convert time window to frame indices
        start_frame = int(response_window_start * frame_rate)
        end_frame = int(response_window_end * frame_rate)
        
        # Extract response window (stimulus onset is at frame pre_frames, so add pre_frames)
        start_idx = pre_frames + start_frame
        end_idx = pre_frames + end_frame
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(cs_plus_trace), end_idx)
        
        # Calculate response in the window based on selected method
        if method == 'peak':
            # Use peak amplitude (maximum value) in the response window
            cs_plus_response = np.max(cs_plus_trace[start_idx:end_idx])
            cs_minus_response = np.max(cs_minus_trace[start_idx:end_idx])
        elif method == 'mean':
            # Use mean fluorescence during the stimulus window
            cs_plus_response = np.mean(cs_plus_trace[start_idx:end_idx])
            cs_minus_response = np.mean(cs_minus_trace[start_idx:end_idx])
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'peak' or 'mean'.")
        
        # Apply negative value clipping if requested
        if set_negative_to_zero:
            cs_plus_response = max(0, cs_plus_response)
            cs_minus_response = max(0, cs_minus_response)
        
        # Calculate selectivity index: (CS+ - CS-) / (CS+ + CS-)
        
        if np.isnan(cs_plus_response) or np.isnan(cs_minus_response):
            selectivity = np.nan
        else:
            numerator = cs_plus_response - cs_minus_response
            denominator = cs_plus_response + cs_minus_response
            
            # Handle division by zero 
            if denominator == 0:
                if numerator > 0:
                    selectivity = np.inf
                elif numerator < 0:
                    selectivity = -np.inf
                else:
                    selectivity = np.nan  # 0/0 case
            else:
                selectivity = numerator / denominator
        
        selectivity_indices[roi_idx] = {
            'selectivity': selectivity,
            'cs_plus_response': cs_plus_response,
            'cs_minus_response': cs_minus_response
        }
    
    return selectivity_indices


def analyze_single_animal_session(data_path, animal_name, session_label, save_path=None, 
                                 max_rois_to_plot=20, response_window_start=0, response_window_end=2,
                                 generate_traces=True, generate_histograms=True, selectivity_method='peak',
                                 set_negative_to_zero=True):
    """
    Analyze CS plus and CS minus responses for a specific animal and session.
    Applies baseline correction by subtracting baseline trace of equivalent duration 
    to stimulus window from stimulus response window on a trial-by-trial basis.
    Calculates selectivity index using either peak amplitude or mean fluorescence in response window.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        animal_name (str): Name of the animal to analyze
        session_label (str): Label of the session to analyze (e.g., 'Pre', 'Day1', 'Post')
        save_path (str, optional): Path to save plots and results
        max_rois_to_plot (int): Maximum number of ROIs to plot
        response_window_start (float): Start of response window for selectivity calculation
        response_window_end (float): End of response window for selectivity calculation
        generate_traces (bool): Whether to generate and plot mean traces
        generate_histograms (bool): Whether to generate selectivity histograms
        selectivity_method (str): Method for selectivity calculation ('peak' or 'mean')
        set_negative_to_zero (bool): If True, set negative df/f values to 0 
        
    Returns:
        dict: Results containing cs_plus_data, cs_minus_data, selectivity_indices, and time_axis
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
    
    # Use all ROIs instead of just tracked ones
    roi_indices = None  # This will use all ROIs in calculate_roi_averaged_traces
    
    # Get session data
    dff = target_session['dff']
    cs_plus_frames = target_session['cs_plus_frames']
    cs_minus_frames = target_session['cs_minus_frames']
    
    # Print first few CS+ and CS- frame ranges for verification
    
    # Check if CS+ and CS- frames are different (they should be!)
    if len(cs_plus_frames) > 0 and len(cs_minus_frames) > 0:
        cs_plus_starts = [frame[0] for frame in cs_plus_frames]
        cs_minus_starts = [frame[0] for frame in cs_minus_frames]
        #print(f"  CS+ start frames: {cs_plus_starts[:5]}...")
        #print(f"  CS- start frames: {cs_minus_starts[:5]}...")
        
        # Check for any overlap (there shouldn't be any)
        overlap = set(cs_plus_starts) & set(cs_minus_starts)
        if overlap:
            print(f"  WARNING: Found overlapping frames between CS+ and CS-: {overlap}")
    
    # Extract trial responses (baseline-corrected for selectivity calculation)
    cs_plus_trials_corrected, time_axis = extract_trial_dff_baseline_corrected(dff, cs_plus_frames)
    cs_minus_trials_corrected, _ = extract_trial_dff_baseline_corrected(dff, cs_minus_frames)
    
    # Extract raw trial responses (for plotting)
    cs_plus_trials_raw, _ = extract_trial_dff(dff, cs_plus_frames)
    cs_minus_trials_raw, _ = extract_trial_dff(dff, cs_minus_frames)
    
    # Calculate ROI-averaged traces (raw for plotting)
    from average_traces import calculate_roi_averaged_traces as calc_raw_traces
    cs_plus_data = calc_raw_traces(cs_plus_trials_raw, roi_indices)
    cs_minus_data = calc_raw_traces(cs_minus_trials_raw, roi_indices)
    
    # Calculate ROI-averaged traces (baseline-corrected for selectivity calculation)
    cs_plus_data_corrected = calculate_roi_averaged_traces(cs_plus_trials_corrected, roi_indices)
    cs_minus_data_corrected = calculate_roi_averaged_traces(cs_minus_trials_corrected, roi_indices)
    
    # Calculate selectivity indices from baseline-corrected data
    selectivity_indices = calculate_selectivity_index(cs_plus_data_corrected, cs_minus_data_corrected,
                                                    response_window_start, response_window_end,
                                                    cs_plus_frames, cs_minus_frames, method=selectivity_method,
                                                    set_negative_to_zero=set_negative_to_zero)
    
    # Generate plots based on user preferences
    if generate_traces:
        # Import here to avoid circular imports
        from average_traces import plot_cs_traces_with_sem
        # Plot traces with selectivity indices
        plot_cs_traces_with_sem(cs_plus_data, cs_minus_data, time_axis, 
                               save_path=save_path, max_rois_per_figure=max_rois_to_plot,
                               animal_name=animal_name, session_label=session_label,
                               selectivity_indices=selectivity_indices)
    
    if generate_histograms:
        # Import here to avoid circular imports
        from selectivity_histogram_singleanimal import print_selectivity_indices, plot_selectivity_histogram
        # Print all selectivity indices (sorted by selectivity)
        print_selectivity_indices(selectivity_indices, animal_name, session_label, sort_by='selectivity')
        
        # Also print sorted by ROI number for reference
        print_selectivity_indices(selectivity_indices, animal_name, session_label, sort_by='roi')
        
        # Plot selectivity histogram
        plot_selectivity_histogram(selectivity_indices, save_path=save_path, 
                                  animal_name=animal_name, session_label=session_label)
    
    # Completed analysis
    
    # Return results for further analysis
    results = {
        'animal_name': animal_name,
        'session_label': session_label,
        'cs_plus_data': cs_plus_data,
        'cs_minus_data': cs_minus_data,
        'selectivity_indices': selectivity_indices,
        'time_axis': time_axis,
        'roi_indices': roi_indices
    }
    
    return results

