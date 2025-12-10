"""
SVM Single-ROI Selectivity Analysis Script

This script tests whether individual ROIs can classify CS+ vs CS- trials.
Unlike the population decoding analysis (svm_decoding_analysis.py), this tests
each ROI independently to identify which neurons are selective. 

We train one decoder per ROI. 

Methodology:
1. Data Loading:
   - Loads processed data from pickle files.
   - Loads animal metadata to identify learners vs. non-learners.
   - Groups sessions into learning phases: Pre, Early Learning, Late Learning, Post.

2. Feature Extraction (per ROI, per trial):
   - Peak Amplitude: (Max dF/F in 3s post-stim) - (Max dF/F in 1s pre-stim)
   - Mean Amplitude: (Mean dF/F in 3s post-stim) - (Mean dF/F in 1s pre-stim)
   - Area Under Curve (AUC): Integral of response during stimulus period
   - Peak × Mean Interaction: Captures response shape (transient vs sustained)
   - Time to Peak: Latency of peak response relative to stimulus onset (in seconds)

3. Decoding Analysis (per ROI, per session):
   - For each ROI individually, train an SVM classifier
   - Classifier: Linear Support Vector Machine (SVM)
   - Optimization: Nested Cross-Validation
     * Outer Loop: 5-fold Stratified CV (for evaluating accuracy)
     * Inner Loop: 3-fold Grid Search CV (for optimizing C parameter)
   - Class Balancing: Majority class subsampled to match minority class count

4. Visualization:
   - Distribution of ROI decoding accuracies across learning phases
   - Points are color-coded by Learner status (Green=Learner, Red=Non-Learner)
   - Each point represents the mean accuracy across all ROIs for one animal

Usage:
   - Ensure 'TEST_MODE' is False for real analysis
   - Run script to generate and save plots to 'save_figures_path'
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# Add data_processing to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(current_dir, '..', 'data_processing')
if data_processing_dir not in sys.path:
    sys.path.append(data_processing_dir)

from data_loader import load_processed_data
import learning_phase_mapping as lpm
from set_paths import processed_data_path, save_figures_path

# Configuration
FRAME_RATE = 30  # Hz
TEST_MODE = False  # Set to True to run sanity check


def extract_roi_features_single_trial(dff_roi, start_frame, baseline_duration=30, stim_duration=90):
    """
    Extract features for a single ROI from a single trial.
    
    Args:
        dff_roi (np.ndarray): dF/F timeseries for one ROI (1D array)
        start_frame (int): Stimulus onset frame
        baseline_duration (int): Number of frames for baseline (default: 30 = 1s)
        stim_duration (int): Number of frames for stimulus period (default: 90 = 3s)
    
    Returns:
        dict: Dictionary with 'peak_amp', 'mean_amp', 'auc', 'peak_mean_interaction'
        None if insufficient data
    """
    # Define windows
    baseline_start = max(0, start_frame - baseline_duration)
    baseline_end = start_frame
    stim_start = start_frame
    stim_end = min(len(dff_roi), start_frame + stim_duration)
    
    # Extract data
    baseline_data = dff_roi[baseline_start:baseline_end]
    stim_data = dff_roi[stim_start:stim_end]
    
    # Check sufficient data
    if len(baseline_data) < baseline_duration or len(stim_data) < stim_duration:
        return None
    
    # Feature 1: Peak Amplitude
    baseline_peak = np.max(baseline_data)
    stim_peak = np.max(stim_data)
    peak_amp = stim_peak - baseline_peak
    
    # Feature 2: Mean Amplitude
    baseline_mean = np.mean(baseline_data)
    stim_mean = np.mean(stim_data)
    mean_amp = stim_mean - baseline_mean
    
    # Feature 3: Area Under Curve (AUC)
    # Baseline-corrected integral
    baseline_corrected_stim = stim_data - baseline_mean
    auc = np.trapz(baseline_corrected_stim)
    
    # Feature 4: Interaction term (peak × mean)
    # Captures response shape: high values = strong sustained response
    peak_mean_interaction = peak_amp * mean_amp
    
    return {
        'peak_amp': peak_amp,
        'mean_amp': mean_amp,
        'auc': auc,
        'peak_mean_interaction': peak_mean_interaction
    }


def extract_single_roi_features(session_data, roi_idx, frame_offset=0):
    """
    Extract features for a single ROI across all trials in a session.
    
    Args:
        session_data (dict): Session data containing dff, cs_plus_frames, cs_minus_frames
        roi_idx (int): Index of the ROI to analyze
        frame_offset (int): Number of frames to shift the analysis window BACKWARDS. 
                            Use 0 for Stimulus, and e.g. 90 for Control (Pre-Stim).
    
    Returns:
        X (np.ndarray): Feature matrix (n_trials, 5)
        y (np.ndarray): Labels (0 for CS-, 1 for CS+)
    """
    dff = session_data['dff']
    dff_roi = dff[roi_idx, :]  # Extract this ROI's timeseries
    
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    
    X = []
    y = []
    
    # Extract CS- trials (Label 0)
    for start_frame, stop_frame in cs_minus_frames:
        adjusted_start = start_frame - frame_offset
        if adjusted_start < 0:
            continue
            
        features = extract_roi_features_single_trial(dff_roi, adjusted_start)
        if features is not None:
            X.append([features['peak_amp'], features['mean_amp'], features['auc'], 
                     features['peak_mean_interaction']])
            y.append(0)
    
    # Extract CS+ trials (Label 1)
    for start_frame, stop_frame in cs_plus_frames:
        adjusted_start = start_frame - frame_offset
        if adjusted_start < 0:
            continue
            
        features = extract_roi_features_single_trial(dff_roi, adjusted_start)
        if features is not None:
            X.append([features['peak_amp'], features['mean_amp'], features['auc'],
                     features['peak_mean_interaction']])
            y.append(1)
    
    if len(X) == 0:
        return None, None
    
    return np.array(X), np.array(y)


def balance_classes(X, y):
    """
    Subsample the majority class to match the minority class count.
    Ensures chance level is 50%.
    """
    n_cs_minus = np.sum(y == 0)
    n_cs_plus = np.sum(y == 1)
    
    if n_cs_minus == 0 or n_cs_plus == 0:
        return None, None
    
    min_samples = min(n_cs_minus, n_cs_plus)
    
    # Separate classes
    X_minus = X[y == 0]
    X_plus = X[y == 1]
    
    # Resample (subsample)
    X_minus_sub = resample(X_minus, replace=False, n_samples=min_samples, random_state=42)
    X_plus_sub = resample(X_plus, replace=False, n_samples=min_samples, random_state=42)
    
    # Combine
    X_balanced = np.vstack((X_minus_sub, X_plus_sub))
    y_balanced = np.hstack((np.zeros(min_samples), np.ones(min_samples)))
    
    return X_balanced, y_balanced


def train_decoder(X, y, cv_folds=5):
    """
    Train SVM decoder using Nested Cross-Validation with Hyperparameter Tuning.
    """
    # If too few samples, reduce folds
    if len(y) < cv_folds * 2:
        cv_folds = max(2, len(y) // 2)
        if cv_folds < 2:
            return np.nan  # Cannot cross-validate
    
    # Define pipeline
    pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    
    # Define parameter grid
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # Setup GridSearch (Inner Loop)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1, scoring='accuracy')
    
    # Setup Outer Loop CV
    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Run Nested CV
    outer_scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner Loop (Grid Search)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        score = best_model.score(X_test, y_test)
        outer_scores.append(score)
    
    return np.mean(outer_scores)


def load_animal_metadata(data_path):
    """
    Load animal metadata from the CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'data_processing', 'animal_metadata.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata CSV not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    metadata = {}
    
    for _, row in df.iterrows():
        animal_id = row['animal_id']
        metadata[animal_id] = {
            'learner': row.get('learner', 'unknown'),
            'cheater': row.get('cheater', 'unknown'),
        }
    
    return metadata


def plot_decoding_results(results, title_suffix, filename_suffix):
    """
    Plots the decoding results (boxplots + individual ROI points).
    """
    phases = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    
    # Prepare data for plotting
    plot_data_scores = []  # List of lists of scores for boxplot
    plot_data_full = []    # List of lists of (score, is_learner) tuples
    
    for phase in phases:
        phase_results = results[phase]
        scores = [r[0] for r in phase_results]
        plot_data_scores.append(scores)
        plot_data_full.append(phase_results)
    
    if not any(plot_data_scores):
        print(f"No valid results to plot for {title_suffix}.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    bp = plt.boxplot(plot_data_scores, labels=phases, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     boxprops=dict(facecolor='lightgray', alpha=0.7),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     showfliers=False)  # Hide outliers to show custom points
    
    # Add individual points colored by learner status
    learner_handle = None
    non_learner_handle = None
    
    for i, phase_data in enumerate(plot_data_full):
        if not phase_data:
            continue
        
        # Unpack
        scores = np.array([x[0] for x in phase_data])
        is_learners = np.array([x[1] for x in phase_data])
        
        # Add jitter to x-coordinates
        x_jitter = np.random.normal(i + 1, 0.06, size=len(scores))
        
        # Plot Learners (Green)
        if np.any(is_learners):
            h1, = plt.plot(x_jitter[is_learners], scores[is_learners], 'o',
                          color='green', alpha=0.4, markersize=5, markeredgewidth=0)
            learner_handle = h1
        
        # Plot Non-Learners (Red)
        if np.any(~is_learners):
            h2, = plt.plot(x_jitter[~is_learners], scores[~is_learners], 'o',
                          color='red', alpha=0.4, markersize=5, markeredgewidth=0)
            non_learner_handle = h2
    
    # Add chance level line
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Chance (50%)')
    
    # Custom legend
    handles = []
    labels = []
    if learner_handle:
        handles.append(learner_handle)
        labels.append('Learner')
    if non_learner_handle:
        handles.append(non_learner_handle)
        labels.append('Non-Learner')
    
    # Add chance line to legend
    h_chance, = plt.plot([], [], color='gray', linestyle='--')
    handles.append(h_chance)
    labels.append('Chance (50%)')
    
    plt.legend(handles, labels, loc='best')
    
    plt.title(f'Single-ROI SVM Decoding Accuracy ({title_suffix})\nFeatures: Peak, Mean, AUC, Peak×Mean\nEach point = ONE ROI', 
              fontsize=12)
    plt.ylabel('Decoding Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save
    save_file = os.path.join(save_figures_path, f'svm_single_roi_{filename_suffix}.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {save_file}")
    plt.show()


def main():
    # 1. Load Data
    print("Loading data...")
    all_data = load_processed_data(processed_data_path)
    
    # Load metadata
    metadata = load_animal_metadata(processed_data_path)
    
    # Container for results: {Phase: [(accuracy, is_learner)]}
    results = {
        'Pre': [],
        'Early Learning': [],
        'Late Learning': [],
        'Post': []
    }
    
    # Container for CONTROL results (Pre-Stimulus)
    results_control = {
        'Pre': [],
        'Early Learning': [],
        'Late Learning': [],
        'Post': []
    }
    
    print(f"Processing {len(all_data)} animals...")
    
    for animal, animal_data in all_data.items():
        print(f"\nAnalyzing {animal}...")
        
        # Determine learner status
        is_learner = False
        if animal in metadata:
            status = metadata[animal].get('learner', 'unknown')
            if status == 'learner':
                is_learner = True
        
        # Iterate through sessions
        sessions = animal_data['sessions']
        for session_idx, session_data in sessions.items():
            label = session_data['label']
            phase = lpm.day_label_to_learning_phase(label)
            
            if phase not in results:
                continue
            
            # Get number of ROIs in this session
            dff = session_data['dff']
            n_rois = dff.shape[0]
            
            print(f"  {label} ({phase}): {n_rois} ROIs")
            
            # Analyze each ROI independently
            roi_accuracies = []
            roi_accuracies_control = []
            
            for roi_idx in range(n_rois):
                # --- 1. ACTUAL STIMULUS DECODING ---
                X, y = extract_single_roi_features(session_data, roi_idx, frame_offset=0)
                
                if X is not None and len(y) >= 10:  # Minimum trials check
                    if TEST_MODE:
                         threshold = np.median(X[:, 0])
                         y = (X[:, 0] > threshold).astype(int)
                    
                    X_bal, y_bal = balance_classes(X, y)
                    
                    if X_bal is not None:
                        accuracy = train_decoder(X_bal, y_bal)
                        
                        if not np.isnan(accuracy):
                            roi_accuracies.append(accuracy)

                # --- 2. CONTROL (PRE-STIMULUS) DECODING ---
                # Shift back 90 frames (3 seconds) to use pre-stim period
                X_ctrl, y_ctrl = extract_single_roi_features(session_data, roi_idx, frame_offset=90)
                
                if X_ctrl is not None and len(y_ctrl) >= 10:
                    if TEST_MODE:
                        # Use random labels for control in test mode
                        y_ctrl = np.random.randint(0, 2, size=len(y_ctrl))
                        
                    X_bal_ctrl, y_bal_ctrl = balance_classes(X_ctrl, y_ctrl)
                    
                    if X_bal_ctrl is not None:
                        accuracy_ctrl = train_decoder(X_bal_ctrl, y_bal_ctrl)
                        
                        if not np.isnan(accuracy_ctrl):
                            roi_accuracies_control.append(accuracy_ctrl)

            # Store individual ROI accuracies
            if len(roi_accuracies) > 0:
                print(f"    Processed {len(roi_accuracies)} ROIs. Mean: {np.mean(roi_accuracies):.3f}")
                for acc in roi_accuracies:
                    results[phase].append((acc, is_learner))
            
            if len(roi_accuracies_control) > 0:
                for acc in roi_accuracies_control:
                    results_control[phase].append((acc, is_learner))
    
    # 2. Plotting
    print("\nPlotting results...")
    
    # Plot 1: Actual Stimulus Window
    plot_decoding_results(results, "CS+ vs CS-", "selectivity_by_phase")
    
    # Plot 2: Control Window (Pre-Stimulus)
    plot_decoding_results(results_control, "Control: Pre-Stimulus Window", "control_pre_stim")


if __name__ == "__main__":
    main()
