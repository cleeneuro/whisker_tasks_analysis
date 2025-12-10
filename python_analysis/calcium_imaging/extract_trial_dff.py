"""
Trial extraction functions for calcium imaging analysis.

This module contains common functions for extracting trial responses from calcium imaging data.
Different extraction methods are provided for different analysis needs:
- extract_trial_dff: Raw dF/F (no baseline correction) - for plotting/visualization
- extract_trial_dff_baseline_corrected: With baseline correction - for selectivity calculations
"""

import numpy as np


def extract_trial_dff(dff, stim_frames, pre_frames=30, post_frames=180, frame_rate=30):
    """
    Extract RAW fluorescence responses for each trial around stimulus presentation.
    This version does NOT apply baseline correction - returns the actual dF/F traces.
    
    This is the standard extraction function used for plotting and visualization.
    For baseline-corrected data (used in selectivity calculations), use 
    extract_trial_dff_baseline_corrected().
    
    Args:
        dff (np.ndarray): dF/F data (ROIs x time)
        stim_frames (list): List of (start_frame, stop_frame) tuples for each trial
        pre_frames (int): Number of frames before stimulus onset (default: 30 frames = 1 second)
        post_frames (int): Number of frames after stimulus onset (default: 180 frames = 6 seconds)
        frame_rate (int): Frame rate in Hz
        
    Returns:
        tuple: (trial_responses, time_axis)
            - trial_responses: Array of shape (n_trials, n_rois, n_frames) - RAW (not baseline corrected)
            - time_axis: Time axis in seconds
    """
    trial_responses = []
    expected_frames = pre_frames + post_frames  # Total expected frames per trial
    
    for start_frame, stop_frame in stim_frames:
        # Define trial window around stimulus onset
        trial_start = max(0, start_frame - pre_frames)
        trial_end = min(dff.shape[1], start_frame + post_frames)
        
        # Extract trial data
        trial_data = dff[:, trial_start:trial_end]  # Shape: (n_rois, n_frames)
        
        # Ensure all trials have the same length by padding or truncating
        if trial_data.shape[1] < expected_frames:
            # Pad with zeros if trial is too short (e.g., at end of recording)
            padding_needed = expected_frames - trial_data.shape[1]
            trial_data = np.pad(trial_data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
        elif trial_data.shape[1] > expected_frames:
            # Truncate if trial is too long (shouldn't happen with current logic, but just in case)
            trial_data = trial_data[:, :expected_frames]
        
        # NO baseline correction - keep raw data
        trial_responses.append(trial_data)
    
    # Convert to array and create time axis
    trial_responses = np.array(trial_responses)  # Shape: (n_trials, n_rois, n_frames)
    n_frames = trial_responses.shape[2]
    time_axis = np.arange(-pre_frames, post_frames) / frame_rate
    
    return trial_responses, time_axis


def extract_trial_dff_baseline_corrected(dff, stim_frames, pre_frames=30, post_frames=180, frame_rate=30):
    """
    Extract fluorescence responses for each trial around stimulus presentation.
    Applies baseline correction by subtracting baseline trace of equivalent duration 
    to stimulus window from stimulus response window on a trial-by-trial basis.
    
    This method is specifically used for selectivity calculations where you want to
    baseline-correct only the stimulus response period using a matched-duration baseline.
    
    Args:
        dff (np.ndarray): dF/F data (ROIs x time)
        stim_frames (list): List of (start_frame, stop_frame) tuples for each trial
        pre_frames (int): Number of frames before stimulus onset (default: 30 frames = 1 second)
        post_frames (int): Number of frames after stimulus onset (default: 180 frames = 6 seconds)
        frame_rate (int): Frame rate in Hz
        
    Returns:
        tuple: (trial_responses, time_axis)
            - trial_responses: Array of shape (n_trials, n_rois, n_frames) - baseline corrected
            - time_axis: Time axis in seconds
    """
    trial_responses = []
    expected_frames = pre_frames + post_frames  # Total expected frames per trial
    
    for start_frame, stop_frame in stim_frames:
        # Calculate stimulus duration in frames
        stimulus_duration = stop_frame - start_frame
        
        # Define trial window around stimulus onset
        trial_start = max(0, start_frame - pre_frames)
        trial_end = min(dff.shape[1], start_frame + post_frames)
        
        # Extract trial data
        trial_data = dff[:, trial_start:trial_end]  # Shape: (n_rois, n_frames)
        
        # Ensure all trials have the same length by padding or truncating
        if trial_data.shape[1] < expected_frames:
            # Pad with zeros if trial is too short (e.g., at end of recording)
            padding_needed = expected_frames - trial_data.shape[1]
            trial_data = np.pad(trial_data, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0)
        elif trial_data.shape[1] > expected_frames:
            # Truncate if trial is too long (shouldn't happen with current logic, but just in case)
            trial_data = trial_data[:, :expected_frames]
        
        # Apply baseline correction for each ROI
        baseline_corrected_trial = np.zeros_like(trial_data)
        
        for roi_idx in range(trial_data.shape[0]):
            # Define baseline period: equivalent duration to stimulus, ending just before stimulus onset
            # The stimulus onset is at frame 'pre_frames' in our trial window
            baseline_end = pre_frames  # At stimulus onset
            baseline_start = max(0, baseline_end - stimulus_duration)  # Equivalent duration before stimulus
            
            # Define stimulus response period: from stimulus onset to stimulus offset
            stim_response_start = pre_frames  # At stimulus onset
            stim_response_end = min(pre_frames + stimulus_duration, trial_data.shape[1])  # At stimulus offset
            
            # Get baseline trace (equivalent duration to stimulus)
            baseline_trace = trial_data[roi_idx, baseline_start:baseline_end]
            
            # Get stimulus response trace
            stim_response_trace = trial_data[roi_idx, stim_response_start:stim_response_end]
            
            # Calculate baseline average
            baseline_avg = np.mean(baseline_trace)
            
            # Apply baseline correction: subtract baseline from stimulus response window
            # Keep the rest of the trial unchanged
            baseline_corrected_trial[roi_idx, :] = trial_data[roi_idx, :].copy()
            baseline_corrected_trial[roi_idx, stim_response_start:stim_response_end] = stim_response_trace - baseline_avg
        
        
        trial_responses.append(baseline_corrected_trial)
    
    # Convert to array and create time axis
    trial_responses = np.array(trial_responses)  # Shape: (n_trials, n_rois, n_frames)
    n_frames = trial_responses.shape[2]
    time_axis = np.arange(-pre_frames, post_frames) / frame_rate
    
    return trial_responses, time_axis


# Aliases for backward compatibility with existing code
extract_trial_responses = extract_trial_dff
extract_raw_trial_responses = extract_trial_dff
extract_trial_responses_baseline_corrected = extract_trial_dff_baseline_corrected

