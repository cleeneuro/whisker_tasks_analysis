"""
Linear Encoding Model Analysis

This script fits a Linear Regression model to explain the fluorescence activity 
of INDIVIDUAL cells using behavioral regressors.

Goal: How much do CS+, CS-, licking and reward modulate neural activity? 

Note: We use 100% of the data for training. Our goal here is to assess the weights,
not to predict the neural activity. 
We want to ask how much do these stimuli modulate the neural activity, therefore we don't do train/test split

Model:
    F(t) = β0 + β1*CS+(t) + β2*CS-(t) + β3*Lick(t) + β4*Reward(t) + ε

Steps:
1. Load processed data.
2. Construct Design Matrix (X) with time binning.
3. Fit model for each neuron across all animals (Late Learning).
4. Visualize population tuning (Scatter plot of weights).
5. Saves a .csv of all ROI weights and R2 score 
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns (not available)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add data_processing to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(current_dir, '..', 'data_processing')
if data_processing_dir not in sys.path:
    sys.path.append(data_processing_dir)

from data_loader import load_processed_data
import learning_phase_mapping as lpm
from set_paths import processed_data_path, save_figures_path

# Configuration
BIN_SIZE_MS = 200  # Time bin size in milliseconds
FRAME_RATE = 30    # Frames per second

def create_design_matrix(session_data, bin_size_ms=200, frame_rate=30):
    """
    Creates the Design Matrix (X) and Target (y) for a single session.
    """
    dff = session_data['dff']
    n_rois, n_frames = dff.shape
    
    # Calculate time vector for original frames
    total_duration_s = n_frames / frame_rate
    original_times = np.linspace(0, total_duration_s, n_frames)
    
    # Define new time bins
    bin_size_s = bin_size_ms / 1000.0
    n_bins = int(np.ceil(total_duration_s / bin_size_s))
    bin_edges = np.linspace(0, total_duration_s, n_bins + 1)
    
    # 1. Bin Neural Data (Y)
    Y_binned = np.zeros((n_bins, n_rois))
    for i in range(n_bins):
        if i == n_bins - 1:
            # Include the last edge for the final bin
            mask = (original_times >= bin_edges[i]) & (original_times <= bin_edges[i+1])
        else:
            mask = (original_times >= bin_edges[i]) & (original_times < bin_edges[i+1])
            
        if np.any(mask):
            Y_binned[i, :] = np.mean(dff[:, mask], axis=1)
            
    # 2. Create Regressors (X)
    regressors = {
        'CS+': np.zeros(n_bins),
        'CS-': np.zeros(n_bins),
        'Lick': np.zeros(n_bins),
        'Reward': np.zeros(n_bins)
    }
    
    def get_bin_idx(t):
        idx = int(t / bin_size_s)
        return min(idx, n_bins - 1)
    
    # --- CS+ and CS- ---
    for stim_type, key in [('CS+', 'cs_plus_frames'), ('CS-', 'cs_minus_frames')]:
        if key in session_data:
            trials = session_data[key]
            for start_frame, stop_frame in trials:
                start_bin = get_bin_idx(start_frame / frame_rate)
                stop_bin = get_bin_idx(stop_frame / frame_rate)
                regressors[stim_type][start_bin:stop_bin+1] = 1.0

    # --- Licks ---
    lick_ts = []
    if 'lick_frames' in session_data:
        lf = session_data['lick_frames']
        if hasattr(lf, '__len__') and len(lf) > 0:
            lick_ts = lf / frame_rate

    for t in lick_ts:
        bin_idx = get_bin_idx(t)
        regressors['Lick'][bin_idx] = 1.0 # Simple binary event

    # --- Rewards (always paired with CS+ stop)---
    if 'cs_plus_frames' in session_data:
        reward_duration_s = 2.0 # the animal takes some time to consume reward
        for start_frame, stop_frame in session_data['cs_plus_frames']:
            reward_start = stop_frame / frame_rate
            reward_end = reward_start + reward_duration_s
            start_bin = get_bin_idx(reward_start)
            stop_bin = get_bin_idx(reward_end)
            regressors['Reward'][start_bin:stop_bin+1] = 1.0
    
    # --- Convolution (account for GCaMP6f kinetics)---
    # We will replace step function regressors with exponential curves with a fast rise and slow decay
    tau = 1.5 
    kernel_duration = 5.0 
    kernel_bins = int(kernel_duration / bin_size_s)
    t_kernel = np.linspace(0, kernel_duration, kernel_bins)
    kernel = np.exp(-t_kernel / tau)
    kernel /= np.sum(kernel) 
    
    # slide kernel across data to convolve 
    for key in list(regressors.keys()):
        convolved = np.convolve(regressors[key], kernel, mode='same')
        regressors[key] = convolved

    X = pd.DataFrame(regressors)
    
    return X, Y_binned, list(X.columns)

def fit_encoding_models(X, Y, feature_names):
    """
    Fit Linear Regression for each ROI.
    Returns: results_df (Weights), scores (R2)
    """
    n_bins, n_rois = Y.shape
    
    scaler_X = StandardScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=feature_names)
    
    weights = []
    scores = []
    model = LinearRegression()
    
    for roi in range(n_rois):
        y_roi = Y[:, roi]
        valid_mask = ~np.isnan(y_roi)
        
        if np.sum(valid_mask) < 10:
            weights.append([np.nan] * len(feature_names))
            scores.append(np.nan)
            continue
        
        # Z-score Y
        y_std = np.std(y_roi[valid_mask]) + 1e-6
        y_scaled = (y_roi[valid_mask] - np.mean(y_roi[valid_mask])) / y_std
        
        model.fit(X_scaled.loc[valid_mask], y_scaled)
        weights.append(model.coef_)
        scores.append(model.score(X_scaled.loc[valid_mask], y_scaled))
        
    results = pd.DataFrame(weights, columns=feature_names)
    results['R2'] = scores
    results['ROI'] = range(n_rois)
    
    return results

def plot_population_weights(all_results_df, save_path, phase_name):
    """
    Boxplot of weights for each regressor across the population.
    Similar style to SVM selectivity plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Filter for valid data
    # Only drop if columns exist
    cols_to_check = ['CS+', 'CS-', 'Lick', 'R2']
    # If Reward was dropped (e.g. Pre session), don't check for it
    df = all_results_df.dropna(subset=[c for c in cols_to_check if c in all_results_df.columns])
    
    # Melt the dataframe for boxplotting
    regressors = ['CS+', 'CS-', 'Lick']
    if 'Reward' in df.columns:
        regressors.append('Reward')
        
    df_melted = df.melt(id_vars=['ROI', 'Animal', 'R2'], 
                        value_vars=[r for r in regressors if r in df.columns], 
                        var_name='Feature', 
                        value_name='Weight')
    
    # Create Boxplot
    boxprops = dict(facecolor='lightgray', alpha=0.7)
    medianprops = dict(color='black', linewidth=1.5)
    
    data_to_plot = []
    labels = []
    valid_regressors = [r for r in regressors if r in df.columns]
    
    for feat in valid_regressors:
        data = df_melted[df_melted['Feature'] == feat]['Weight'].values
        data_to_plot.append(data)
        labels.append(feat)
        
    if not data_to_plot:
        print(f"No data to plot for {phase_name}")
        return

    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     medianprops=medianprops, boxprops=boxprops,
                     showfliers=False) 
                     
    # Overlay individual points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(i + 1, 0.08, size=len(data))
        r2_vals = df_melted[df_melted['Feature'] == labels[i]]['R2'].values
        plt.scatter(x, data, c=r2_vals, cmap='viridis', alpha=0.6, s=15, zorder=2)
        
    plt.colorbar(label='Model R2')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'Population Encoding Weights (Standardized Beta)\n{phase_name}', fontsize=16)
    plt.ylabel('Weight (std)', fontsize=14)
    plt.xlabel('Regressor', fontsize=14)
    plt.ylim(-0.4, 0.4)
    
    # Add n count once in bottom right
    total_n = len(data_to_plot[0]) if data_to_plot else 0
    plt.text(0.98, 0.02, f'n={total_n} ROIs', 
             transform=plt.gca().transAxes, 
             ha='right', va='bottom', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    filename = f'population_encoding_weights_{phase_name.replace(" ", "_")}.png'
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    print(f"Population Boxplot saved to {save_file}")

def analyze_phase(all_data, phase_name):
    """Helper to analyze all animals for a specific phase"""
    phase_results = []
    print(f"\nProcessing {phase_name}...")
    
    for animal_name, animal_data in all_data.items():
        target_session = None
        
        for session_key, session in animal_data['sessions'].items():
            label = session['label']
            phase = lpm.day_label_to_learning_phase(label)
            
            if phase == phase_name:
                # Check licks
                if 'lick_frames' in session and hasattr(session['lick_frames'], '__len__') and len(session['lick_frames']) > 10:
                     if np.max(session['lick_frames']) > 0:
                         target_session = session
                         break
        
        if target_session is None:
            # print(f"  Skipping {animal_name}: No valid {phase_name} session with licks found.")
            continue
            
        print(f"  Analyzing {animal_name} - {target_session['label']}")
        
        try:
            X, Y, features = create_design_matrix(target_session, BIN_SIZE_MS, FRAME_RATE)
            results = fit_encoding_models(X, Y, features)
            results['Animal'] = animal_name
            phase_results.append(results)
            
        except Exception as e:
            print(f"    Error analyzing {animal_name}: {e}")
            
    if phase_results:
        return pd.concat(phase_results, ignore_index=True)
    else:
        return None

def main():
    print("Loading data...")
    try:
        all_data = load_processed_data(processed_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Analyze Early Learning
    early_df = analyze_phase(all_data, 'Early Learning')
    
    # Analyze Late Learning
    late_df = analyze_phase(all_data, 'Late Learning')
    
    if save_figures_path:
        regression_path = os.path.join(save_figures_path, 'regression')
        os.makedirs(regression_path, exist_ok=True)
        
        if early_df is not None:
            print(f"Early Learning: {len(early_df)} cells")
            plot_population_weights(early_df, regression_path, 'Early Learning')
            early_df.to_csv(os.path.join(regression_path, 'encoding_weights_early.csv'), index=False)
            
        if late_df is not None:
            print(f"Late Learning: {len(late_df)} cells")
            plot_population_weights(late_df, regression_path, 'Late Learning')
            late_df.to_csv(os.path.join(regression_path, 'encoding_weights_late.csv'), index=False)

if __name__ == "__main__":
    main()
