"""
SVM Decoding Analysis Script

This script loads processed calcium imaging data and trains an SVM to classify 
CS+ vs CS- trials for each session.

THIS CHECKS FOR POPULATION CODING -  a single example is trial i and each ROI is a feature. 
This asks, on a trial to trial basis can the population of ROIs decode CS+ and CS-

Methodology:
1. Data Loading:
   - Loads processed data from pickle files.
   - Loads animal metadata to identify learners vs. non-learners.
   - Groups sessions into learning phases: Pre, Early Learning, Late Learning, Post.

2. Feature Extraction:
   - Uses 'Peak Response Amplitude' for each ROI.
   - Peak Response = (Max dF/F in 3s post-stim) - (Max dF/F in 1s pre-stim).
   - This matches the logic in 'histogram_peak_responses.py'.

3. Decoding Analysis (per session):
   - Classifier: Linear Support Vector Machine (SVM).
   - Optimization: Nested Cross-Validation.
     * Outer Loop: 5-fold Stratified CV (for evaluating accuracy).
     * Inner Loop: 3-fold Grid Search CV (for optimizing C parameter).
   - Class Balancing: Majority class is subsampled to match minority class count (50% chance level).

4. Visualization:
   - Plots decoding accuracy across learning phases.
   - Points are color-coded by Learner status (Green=Learner, Red=Non-Learner).
   - Includes boxplots to show distribution across all animals.

Usage:
   - Ensure 'TEST_MODE' is False for real analysis.
   - Run script to generate and save plots to 'save_figures_path'.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
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

# Import the feature extraction function directly to avoid duplication
from histogram_peak_responses import extract_peak_amplitudes

# Configuration
FRAME_RATE = 30  # Hz
TEST_MODE = False # Set to True to run sanity check (fake labels based on data)

def extract_session_features(session_data):
    """
    Extract features for SVM decoding from a single session.
    
    Features: Peak Response Amplitude (Stim Peak - Baseline Peak) for each ROI.
    Method imported from histogram_peak_responses.py
    
    Returns:
        X (np.ndarray): Feature matrix (n_trials, n_rois)
        y (np.ndarray): Labels (0 for CS-, 1 for CS+)
    """
    dff = session_data['dff']
    cs_plus_frames = session_data['cs_plus_frames']
    cs_minus_frames = session_data['cs_minus_frames']
    
    X = []
    y = []
    
    # Extract CS- trials (Label 0)
    # extract_peak_amplitudes returns a dict: {trial_idx: {'max_amplitudes': ...}}
    cs_minus_peaks_dict = extract_peak_amplitudes(dff, cs_minus_frames, FRAME_RATE)
    for trial_data in cs_minus_peaks_dict.values():
        X.append(trial_data['max_amplitudes'])
        y.append(0)
        
    # Extract CS+ trials (Label 1)
    cs_plus_peaks_dict = extract_peak_amplitudes(dff, cs_plus_frames, FRAME_RATE)
    for trial_data in cs_plus_peaks_dict.values():
        X.append(trial_data['max_amplitudes'])
        y.append(1)
    
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
    
    Inner Loop: Optimizes 'C' parameter (regularization).
    Outer Loop: Estimates generalization error.
    """
    # If too few samples, reduce folds
    if len(y) < cv_folds * 2:
        cv_folds = max(2, len(y) // 2)
        if cv_folds < 2:
            return np.nan # Cannot cross-validate
            
    # Define pipeline
    # Note: We apply StandardScaler inside the loop to avoid data leakage
    pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    
    # Define parameter grid to search
    # C: Regularization parameter. 
    # Smaller C = Stronger regularization (simple boundary, high bias)
    # Larger C = Weaker regularization (complex boundary, high variance)
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # Setup GridSearch (Inner Loop)
    # We use a simpler CV for the inner loop (3-fold) to save time
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1, scoring='accuracy')
    
    # Setup Outer Loop CV
    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Run Nested CV Manually to inspect parameters
    outer_scores = []
    best_params = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner Loop (Grid Search)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_C = grid_search.best_params_['svc__C']
        best_params.append(best_C)
        
        # Evaluate on test set
        score = best_model.score(X_test, y_test)
        outer_scores.append(score)
        
    # Print average C (or mode) to give user feedback
    # Note: Printing inside function is a side effect, but requested by user
    print(f"    Best C values: {best_params}")
    
    return np.mean(outer_scores)

def load_animal_metadata(data_path):
    """
    Load animal metadata from the CSV file.
    Adapted from selectivity_histogram_pooled.py
    """
    # Get the directory of the data processing module to find the CSV file
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

def main():
    # 1. Load Data
    print("Loading data...")
    all_data = load_processed_data(processed_data_path)
    
    # Load metadata
    metadata = load_animal_metadata(processed_data_path)
    
    # Container for results: {Phase: [scores]}
    # We need to store more info now: (score, animal_id, learner_status)
    results = {
        'Pre': [],
        'Early Learning': [],
        'Late Learning': [],
        'Post': []
    }
    
    # Track animal names for detailed inspection if needed
    animal_results = {} 

    print(f"Processing {len(all_data)} animals...")
    
    for animal, animal_data in all_data.items():
        print(f"\nAnalyzing {animal}...")
        
        # Determine learner status
        is_learner = False
        if animal in metadata:
            status = metadata[animal].get('learner', 'unknown')
            if status == 'learner':
                is_learner = True
        
        # Get phase mapping for this animal
        phase_mapping = lpm.convert_session_labels_to_phases(animal_data)
        
        animal_phase_scores = {k: [] for k in results.keys()}
        
        # Iterate ALL sessions to ensure we capture multiple sessions per phase if they exist
        sessions = animal_data['sessions']
        for session_idx, session_data in sessions.items():
            label = session_data['label']
            phase = lpm.day_label_to_learning_phase(label)
            
            if phase not in results:
                continue
                
            # Extract features
            X, y = extract_session_features(session_data)
            
            if X is None or len(y) < 10: # Minimum trials check
                print(f"  Skipping {label} ({phase}): Insufficient data")
                continue
                
            # SANITY CHECK: Overwrite labels if in TEST_MODE
            if TEST_MODE:
                # Create a fake rule: Label 1 if Neuron 0 activity > Median, else 0
                # This creates a perfectly separable dataset (mostly) based on Neuron 0
                neuron_0_activity = X[:, 0]
                threshold = np.median(neuron_0_activity)
                y = (neuron_0_activity > threshold).astype(int)
                print(f"  [TEST MODE] Overwrote labels based on Neuron 0 activity")
                
            # Balance classes
            X_bal, y_bal = balance_classes(X, y)
            if X_bal is None:
                print(f"  Skipping {label} ({phase}): Class imbalance/missing")
                continue
                
            # Train
            score = train_decoder(X_bal, y_bal)
            
            if np.isnan(score):
                print(f"  Skipping {label} ({phase}): CV failed")
                continue
                
            print(f"  {label} ({phase}): Accuracy = {score:.2f} (n={len(y_bal)})")
            
            # Store
            animal_phase_scores[phase].append(score)
            
        # Aggregate per animal (mean of sessions in that phase) to contribute one point per animal
        for phase, scores in animal_phase_scores.items():
            if scores:
                avg_score = np.mean(scores)
                
                # Store tuple: (score, is_learner)
                results[phase].append((avg_score, is_learner))
                
                if animal not in animal_results:
                    animal_results[animal] = {}
                animal_results[animal][phase] = avg_score

    # 2. Plotting
    print("\nPlotting results...")
    phases = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    
    # Prepare data for plotting
    plot_data_scores = [] # List of lists of scores for boxplot
    plot_data_full = []   # List of lists of (score, is_learner) tuples
    
    for phase in phases:
        phase_results = results[phase]
        scores = [r[0] for r in phase_results]
        plot_data_scores.append(scores)
        plot_data_full.append(phase_results)
            
    if not any(plot_data_scores):
        print("No valid results to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot using standard matplotlib
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
        x_jitter = np.random.normal(i + 1, 0.04, size=len(scores))
        
        # Plot Learners (Green)
        if np.any(is_learners):
            h1, = plt.plot(x_jitter[is_learners], scores[is_learners], 'o', 
                     color='green', alpha=0.7, markersize=6, markeredgecolor='black')
            learner_handle = h1
            
        # Plot Non-Learners (Red/Orange)
        if np.any(~is_learners):
            h2, = plt.plot(x_jitter[~is_learners], scores[~is_learners], 'o', 
                     color='red', alpha=0.7, markersize=6, markeredgecolor='black')
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
    h_chance, = plt.plot([], [], color='gray', linestyle='--', label='Chance (50%)')
    handles.append(h_chance)
    labels.append('Chance (50%)')
    
    plt.legend(handles, labels, loc='best')
    
    plt.title('SVM Decoding Accuracy (CS+ vs CS-)\nFeature: Peak Response Amplitude (Stim - Baseline)', fontsize=14)
    plt.ylabel('Decoding Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save
    save_file = os.path.join(save_figures_path, 'svm_decoding_accuracy_by_learner_status.png')
    plt.savefig(save_file, dpi=300)
    print(f"Figure saved to {save_file}")
    plt.show()

if __name__ == "__main__":
    main()
