"""
This script test for significance of an ROI's stimulus resposnes within a session.
The % of significant ROIs for both the mean and peak dff are saved to a .txt file

It calculates both the mean and peak dff for the baseline and stimulus periods for each ROI.
It then runs signed rank tests (mean and peak separately) to test for significance of the stimulus responses.
It saves the results to a .txt file 

Input:
    - data_path: Path to the pickle file containing processed data
    - output_dir: Path to the directory where the results will be saved
    - baseline_duration_seconds: Duration of baseline period in seconds
    - stimulus_duration_seconds: Duration of stimulus period in seconds

Output:
    - Log file with the results for each ROI
"""
import numpy as np
import pickle
from scipy.stats import wilcoxon
import os
import sys
from datetime import datetime

# Add the data_processing directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)
from set_paths import processed_data_path, save_figures_path  # noqa: E402
from data_loader import load_processed_data  # noqa: E402

def extract_baseline_and_stimulus_data(dff, stim_frames, frame_rate=30, baseline_duration_seconds=1.5, stimulus_duration_seconds=3):
    """
    Extract baseline and stimulus period data.
    
    Args:
        dff (np.ndarray): dF/F data (ROIs x frames)
        stim_frames (list): List of (start_frame, stop_frame) tuples for each trial
        frame_rate (int): Frame rate in Hz
        baseline_duration_seconds (float): Duration of baseline period in seconds
        stimulus_duration_seconds (float): Duration of stimulus period in seconds
        
    Returns:
        tuple: (baseline_data, stimulus_data)
            - baseline_data: List of arrays, each with shape (n_rois, baseline_frames) for each trial
            - stimulus_data: List of arrays, each with shape (n_rois, stimulus_frames) for each trial
    """
    baseline_duration_frames = int(baseline_duration_seconds * frame_rate)
    baseline_data = []
    stimulus_data = []
    stimulus_duration_frames = int(stimulus_duration_seconds * frame_rate)
    
    for start_frame, stop_frame in stim_frames:
        # Extract baseline period: 1.5s before stimulus, not including stimulus
        baseline_start = max(0, start_frame - baseline_duration_frames)
        baseline_end = start_frame  # Up to but not including stimulus
        
        # Extract stimulus period
        stimulus_start = start_frame
        stimulus_end = start_frame + stimulus_duration_frames
        
        # Get the data for this trial
        if baseline_start < baseline_end and baseline_end <= dff.shape[1]:
            baseline_trial = dff[:, baseline_start:baseline_end]
            baseline_data.append(baseline_trial)
        
        if stimulus_start < stimulus_end and stimulus_end <= dff.shape[1]:
            stimulus_trial = dff[:, stimulus_start:stimulus_end]
            stimulus_data.append(stimulus_trial)
    
    return baseline_data, stimulus_data

def calculate_roi_stats(baseline_data, stimulus_data):
    """
    Calculate mean and peak dff for baseline and stimulus periods for each ROI.
    
    Args:
        baseline_data: List of arrays with baseline data for each trial
        stimulus_data: List of arrays with stimulus data for each trial
        
    Returns:
        dict: Statistics for each ROI
    """
    if not baseline_data or not stimulus_data:
        return {}
    
    n_rois = baseline_data[0].shape[0]
    roi_stats = {}
    
    for roi_idx in range(n_rois):
        # Collect all baseline and stimulus values for this ROI across trials
        baseline_means = []
        baseline_peaks = []
        stimulus_means = []
        stimulus_peaks = []
        
        # get mean and peak of baseline data
        for trial_baseline in baseline_data:
            if roi_idx < trial_baseline.shape[0]:
                roi_baseline = trial_baseline[roi_idx, :]
                baseline_means.append(np.mean(roi_baseline))
                baseline_peaks.append(np.max(roi_baseline))
        
        # get mean and peak of stimulus data
        for trial_stimulus in stimulus_data:
            if roi_idx < trial_stimulus.shape[0]:
                roi_stimulus = trial_stimulus[roi_idx, :]
                stimulus_means.append(np.mean(roi_stimulus))
                stimulus_peaks.append(np.max(roi_stimulus))
        
        roi_stats[roi_idx] = {
            'baseline_means': np.array(baseline_means),
            'baseline_peaks': np.array(baseline_peaks),
            'stimulus_means': np.array(stimulus_means),
            'stimulus_peaks': np.array(stimulus_peaks)
        }
    
    return roi_stats

def run_statistical_tests(roi_stats):
    """
    Run signed rank tests for each ROI comparing baseline vs stimulus.
    
    Args:
        roi_stats: Dictionary with ROI statistics
        
    Returns:
        dict: Statistical test results for each ROI
    """
    test_results = {}
    
    for roi_idx, stats in roi_stats.items():
        baseline_means = stats['baseline_means']
        baseline_peaks = stats['baseline_peaks']
        stimulus_means = stats['stimulus_means']
        stimulus_peaks = stats['stimulus_peaks']
        
        results = {}
        
        # Test 1: Compare mean dff during baseline vs stimulus
        if len(baseline_means) > 0 and len(stimulus_means) > 0:
            min_trials = min(len(baseline_means), len(stimulus_means))
            try:
                stat_mean, p_mean = wilcoxon(baseline_means[:min_trials], 
                                           stimulus_means[:min_trials])
                results['mean_test'] = {'statistic': stat_mean, 'p_value': p_mean}
            except ValueError:
                results['mean_test'] = {'statistic': np.nan, 'p_value': np.nan}
        else:
            results['mean_test'] = {'statistic': np.nan, 'p_value': np.nan}
        
        # Test 2: Compare peak dff during baseline vs stimulus
        if len(baseline_peaks) > 0 and len(stimulus_peaks) > 0:
            min_trials = min(len(baseline_peaks), len(stimulus_peaks))
            try:
                stat_peak, p_peak = wilcoxon(baseline_peaks[:min_trials], 
                                           stimulus_peaks[:min_trials])
                results['peak_test'] = {'statistic': stat_peak, 'p_value': p_peak}
            except ValueError:
                results['peak_test'] = {'statistic': np.nan, 'p_value': np.nan}
        else:
            results['peak_test'] = {'statistic': np.nan, 'p_value': np.nan}
        
        test_results[roi_idx] = results
    
    return test_results

def analyze_session(animal_name, session_data, session_label, baseline_duration_seconds=1.5, stimulus_duration_seconds=3):
    """
    Analyze a single session for stimulus responses.
    
    Args:
        animal_name: Name of the animal
        session_data: Session data dictionary
        session_label: Label of the session
        baseline_duration_seconds (float): Duration of baseline period in seconds
        stimulus_duration_seconds (float): Duration of stimulus period in seconds
        
    Returns:
        dict: Analysis results
    """
    
    print(f"Analyzing {animal_name} - {session_label}-----------------")
    
    dff = session_data['dff']
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    
    #print(f"  CS+ trials: {len(cs_plus_frames)}, CS- trials: {len(cs_minus_frames)}")
    
    results = {'cs_plus': {}, 'cs_minus': {}}
    
    # Analyze CS+ trials
    if len(cs_plus_frames) > 0:
        baseline_data, stimulus_data = extract_baseline_and_stimulus_data(dff, cs_plus_frames, 
                                                                         baseline_duration_seconds=baseline_duration_seconds, 
                                                                         stimulus_duration_seconds=stimulus_duration_seconds)
        roi_stats = calculate_roi_stats(baseline_data, stimulus_data)
        test_results = run_statistical_tests(roi_stats)
        results['cs_plus'] = test_results
    
    # Analyze CS- trials
    if len(cs_minus_frames) > 0:
        baseline_data, stimulus_data = extract_baseline_and_stimulus_data(dff, cs_minus_frames, 
                                                                         baseline_duration_seconds=baseline_duration_seconds, 
                                                                         stimulus_duration_seconds=stimulus_duration_seconds)
        roi_stats = calculate_roi_stats(baseline_data, stimulus_data)
        test_results = run_statistical_tests(roi_stats)
        results['cs_minus'] = test_results
    
    return results

def count_significant_rois(test_results, alpha=0.05):
    """
    Count the number of statistically significant ROIs.
    
    Args:
        test_results: Dictionary with test results
        alpha: Significance threshold
        
    Returns:
        dict: Count of significant ROIs for each test type
    """
    counts = {
        'mean_test_significant': 0,
        'peak_test_significant': 0,
        'total_rois': len(test_results)
    }
    
    for roi_idx, results in test_results.items():
        if results['mean_test']['p_value'] < alpha:
            counts['mean_test_significant'] += 1
        if results['peak_test']['p_value'] < alpha:
            counts['peak_test_significant'] += 1
    
    return counts

def main():
    # Define analysis parameters
    baseline_duration_seconds = 1.5
    stimulus_duration_seconds = 3
    
    # Create output directory for log files
    output_dir = os.path.join(save_figures_path, 'significance_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped log file
    log_file = os.path.join(output_dir, f"stimulus_response_stats.txt")
    
    # Redirect stdout to log file
    original_stdout = sys.stdout
    with open(log_file, 'w') as f:
        sys.stdout = f
        
        print(f"Stimulus Response Statistics Analysis")
        print("=" * 60)
        print(f"baseline_duration_seconds = {baseline_duration_seconds}")
        print(f"stimulus_duration_seconds = {stimulus_duration_seconds}")
        print()
        
        # Set data path
        data_path = processed_data_path
        
        # Load data
        all_data = load_processed_data(data_path)
    
        # Process each animal and session
        for animal_name, animal_data in all_data.items():
            print(f"\n=== Processing {animal_name} ===")
            
            sessions = animal_data['sessions']
            
            for session_idx, session_data in sessions.items():
                session_label = session_data['label']
                
                # Analyze this session
                results = analyze_session(animal_name, session_data, session_label, 
                                       baseline_duration_seconds, stimulus_duration_seconds)
                
                # initialize variables for organized printing
                cs_plus_mean_percent = 0
                cs_plus_peak_percent = 0
                cs_minus_mean_percent = 0
                cs_minus_peak_percent = 0
                cs_plus_total_rois = 0
                cs_plus_mean_significant = 0
                cs_plus_peak_significant = 0
                cs_minus_total_rois = 0
                cs_minus_mean_significant = 0
                cs_minus_peak_significant = 0
                
                # Get CS+ results
                if results['cs_plus']:
                    cs_plus_counts = count_significant_rois(results['cs_plus'])
                    cs_plus_mean_percent = (cs_plus_counts['mean_test_significant'] / cs_plus_counts['total_rois'])*100
                    cs_plus_peak_percent = (cs_plus_counts['peak_test_significant'] / cs_plus_counts['total_rois'])*100
                    cs_plus_total_rois = cs_plus_counts['total_rois']
                    cs_plus_mean_significant = cs_plus_counts['mean_test_significant']
                    cs_plus_peak_significant = cs_plus_counts['peak_test_significant']
                
                # Get CS- results
                if results['cs_minus']:
                    cs_minus_counts = count_significant_rois(results['cs_minus'])
                    cs_minus_mean_percent = (cs_minus_counts['mean_test_significant'] / cs_minus_counts['total_rois'])*100
                    cs_minus_peak_percent = (cs_minus_counts['peak_test_significant'] / cs_minus_counts['total_rois'])*100
                    cs_minus_total_rois = cs_minus_counts['total_rois']
                    cs_minus_mean_significant = cs_minus_counts['mean_test_significant']
                    cs_minus_peak_significant = cs_minus_counts['peak_test_significant']
                
                # Print results organized by analysis type
                print(f"\n  Results using MEAN:")
                if cs_plus_total_rois > 0:
                    print(f"    CS+: {cs_plus_mean_significant}/{cs_plus_total_rois}, {cs_plus_mean_percent:.2f}% significant ROIs")
                if cs_minus_total_rois > 0:
                    print(f"    CS-: {cs_minus_mean_significant}/{cs_minus_total_rois}, {cs_minus_mean_percent:.2f}% significant ROIs")
                
                print(f"\n  Results using PEAK:")
                if cs_plus_total_rois > 0:
                    print(f"    CS+: {cs_plus_peak_significant}/{cs_plus_total_rois}, {cs_plus_peak_percent:.2f}% significant ROIs")
                if cs_minus_total_rois > 0:
                    print(f"    CS-: {cs_minus_peak_significant}/{cs_minus_total_rois}, {cs_minus_peak_percent:.2f}% significant ROIs")
                
                print()  # Add blank line after results
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Analysis complete! Results saved to: {log_file}")

if __name__ == "__main__":
    main()
