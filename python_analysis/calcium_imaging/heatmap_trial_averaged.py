"""
Trial-Averaged Heatmap Generator

This script creates heatmaps showing trial-averaged dF/F responses for all ROIs.
Each ROI is averaged across all trials within a session.
- X-axis: Time (matching plot_individual_trial_traces limits)
- Y-axis: ROI number
- Each row shows one ROI's averaged response
- Subplots: CS+ (left column) vs CS- (right column)
- Rows: Pre, Early Learning, Late Learning, Post (based on learning_phase_mapping)

Usage:
    python trial_averaged_heatmap.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm

# Import shared data loading functions
from data_loader import load_processed_data
from extract_trial_dff import extract_trial_dff

# Add the data_processing directory to Python path for the import
current_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_dir = os.path.join(os.path.dirname(current_dir), 'data_processing')
sys.path.insert(0, data_processing_dir)

from learning_phase_mapping import convert_animal_data_to_phases  # noqa: E402
from set_paths import processed_data_path, save_figures_path  # noqa: E402


def calculate_trial_averaged_responses(trial_responses):
    """
    Calculate trial-averaged responses for each ROI.
    
    Args:
        trial_responses (np.ndarray): Array of shape (n_trials, n_rois, n_frames)
        
    Returns:
        np.ndarray: Array of shape (n_rois, n_frames) - averaged across trials
    """
    return np.mean(trial_responses, axis=0)  # Average across trials (axis=0)


def find_peak_timing_for_rois(trial_averaged_responses, time_axis):
    """
    Find the peak timing for each ROI based on the maximum amplitude during the plot duration.
    
    Args:
        trial_averaged_responses (np.ndarray): Array of shape (n_rois, n_frames) - averaged responses
        time_axis (np.ndarray): Time axis in seconds
        
    Returns:
        np.ndarray: Array of peak time indices for each ROI (earlier peaks have smaller indices)
    """
    n_rois, n_frames = trial_averaged_responses.shape
    peak_timings = np.zeros(n_rois, dtype=int)
    
    for roi_idx in range(n_rois):
        roi_trace = trial_averaged_responses[roi_idx, :]
        
        # Find the frame with maximum absolute amplitude
        # This handles both positive and negative peaks
        peak_frame = np.argmax(np.abs(roi_trace))
        peak_timings[roi_idx] = peak_frame
    
    return peak_timings


def sort_rois_by_peak_timing(trial_averaged_responses, time_axis):
    """
    Sort ROIs by their peak timing. Later peaks get higher y positions (lower indices).
    
    Args:
        trial_averaged_responses (np.ndarray): Array of shape (n_rois, n_frames) - averaged responses
        time_axis (np.ndarray): Time axis in seconds
        
    Returns:
        tuple: (sorted_responses, roi_order)
            - sorted_responses: Array of shape (n_rois, n_frames) with ROIs reordered
            - roi_order: Array of original ROI indices in the new order
    """
    peak_timings = find_peak_timing_for_rois(trial_averaged_responses, time_axis)
    
    # Sort by peak timing (later peaks first - inverted order)
    # np.argsort gives indices that would sort the array, then reverse for inverted order
    roi_order = np.argsort(peak_timings)[::-1]
    
    # Reorder the responses according to the new ROI order
    sorted_responses = trial_averaged_responses[roi_order, :]
    
    return sorted_responses, roi_order


def create_trial_averaged_heatmap(animal_data, animal_name, save_path=None, figsize=(6, 12)):
    """
    Create trial-averaged heatmap for all ROIs across learning phases.
    
    Args:
        animal_data (dict): Animal data dictionary
        animal_name (str): Name of the animal
        save_path (str, optional): Path to save the heatmap
        figsize (tuple): Figure size
    """
    # Convert day labels to learning phases and get ordered labels
    session_mapping, session_labels = convert_animal_data_to_phases(animal_data)
    n_sessions = len(session_labels)
    
    print(f"Creating heatmap for {animal_name}")
    print(f"Available sessions: {session_labels}")
    
    # Create subplots with dynamic number of rows based on available sessions
    # 2 columns: CS+ (left), CS- (right)
    # Make figure more square by adjusting height calculation
    height_per_session = figsize[1] / max(4, n_sessions)  # Limit height growth for more square plots
    fig, axes = plt.subplots(n_sessions, 2, figsize=(figsize[0], height_per_session * n_sessions))
    
    # Handle case where there's only one session (axes won't be 2D)
    if n_sessions == 1:
        axes = axes.reshape(1, -1)
    
    # Add overall title at the top with more space
    fig.suptitle(f'{animal_name} - Trial-Averaged Heatmap (All ROIs)', fontsize=16, fontweight='bold', y=0.96)
    
    # Add column headers below the main title with more space
    # CS+ header (left column)
    axes[0, 0].text(0.5, 1.20, 'CS+', transform=axes[0, 0].transAxes, 
                    fontsize=14, fontweight='bold', ha='center', va='bottom')
    # CS- header (right column)  
    axes[0, 1].text(0.5, 1.20, 'CS-', transform=axes[0, 1].transAxes,
                    fontsize=14, fontweight='bold', ha='center', va='bottom')
    
    # Track the maximum number of ROIs across all sessions for consistent y-axis
    max_rois = 0
    session_data_cache = {}
    global_norm = None  # Store normalization for colorbar
    
    # First pass: extract all data and find max ROIs
    for plot_idx, session_label in enumerate(session_labels):
        session_idx = session_mapping[session_label]
        
        if session_idx not in animal_data['sessions']:
            print(f"  Warning: Session {session_idx} ({session_label}) not found in data")
            session_data_cache[session_label] = None
            continue
            
        session_data = animal_data['sessions'][session_idx]
        dff = session_data['dff']
        cs_plus_frames = session_data['cs_plus_frames']
        cs_minus_frames = session_data['cs_minus_frames']
        
        print(f"  Processing {session_label}: {dff.shape[0]} ROIs, {len(cs_plus_frames)} CS+ trials, {len(cs_minus_frames)} CS- trials")
        
        # Extract trial responses
        cs_plus_trials, time_axis = extract_trial_dff(dff, cs_plus_frames)
        cs_minus_trials, _ = extract_trial_dff(dff, cs_minus_frames)
        
        # Calculate trial-averaged responses
        cs_plus_avg = calculate_trial_averaged_responses(cs_plus_trials)  # Shape: (n_rois, n_frames)
        cs_minus_avg = calculate_trial_averaged_responses(cs_minus_trials)  # Shape: (n_rois, n_frames)
        
        # Sort ROIs by peak timing for CS+ trials (earlier peaks = higher y position)
        cs_plus_sorted, roi_order = sort_rois_by_peak_timing(cs_plus_avg, time_axis)
        
        # Apply the same ROI order to CS- trials
        cs_minus_sorted = cs_minus_avg[roi_order, :]
        
        print(f"    ROI sorting for {session_label}: {roi_order[:5]}... (showing first 5 ROIs)")
        print(f"    Peak timings (frames): {find_peak_timing_for_rois(cs_plus_avg, time_axis)[roi_order[:5]]}")
        
        session_data_cache[session_label] = {
            'cs_plus_avg': cs_plus_sorted,
            'cs_minus_avg': cs_minus_sorted,
            'time_axis': time_axis,
            'n_rois': dff.shape[0],
            'roi_order': roi_order
        }
        
        max_rois = max(max_rois, dff.shape[0])
    
    print(f"  Maximum ROIs across sessions: {max_rois}")
    
    # Second pass: create heatmaps
    for plot_idx, session_label in enumerate(session_labels):
        # Get axes for CS+ (left column) and CS- (right column)
        ax_plus = axes[plot_idx, 0]  # Left column for CS+
        ax_minus = axes[plot_idx, 1]  # Right column for CS-
        
        cached_data = session_data_cache[session_label]
        
        if cached_data is None:
            # No data available - leave subplots empty but labeled
            for ax in [ax_plus, ax_minus]:
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('ROI')
                ax.set_yticks(range(0, max_rois, max(1, max_rois//10)))  # Show ROI ticks
                ax.set_xlim(-1.0, 4)  # Match plot_individual_trial_traces limits
                ax.set_ylim(0, max_rois)
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
            
            # Add session label on the left with more spacing
            ax_plus.text(-0.35, 0.5, f'{session_label}', transform=ax_plus.transAxes, 
                        fontsize=12, fontweight='bold', rotation=90, 
                        verticalalignment='center', horizontalalignment='center')
            continue
        
        cs_plus_avg = cached_data['cs_plus_avg']
        cs_minus_avg = cached_data['cs_minus_avg']
        time_axis = cached_data['time_axis']
        n_rois = cached_data['n_rois']
        roi_order = cached_data['roi_order']
        
        # Create heatmaps with fixed offset colormap (matching heatmap_single_trial)
        # Use a fixed range to match individual trial heatmaps
        vmin_fixed = -0.05  # Fixed minimum value (matching heatmap_single_trial)
        vmax_fixed = 0.2   # Fixed maximum value (matching heatmap_single_trial)
        vcenter_fixed = vmin_fixed + (vmax_fixed - vmin_fixed) / 2  # Center value
        
        # Create normalization that centers the colormap
        norm = TwoSlopeNorm(vmin=vmin_fixed, vcenter=vcenter_fixed, vmax=vmax_fixed)
        
        # Store the normalization for the colorbar (use the first session's normalization)
        if global_norm is None:
            global_norm = norm
        
        # CS+ heatmap
        im_plus = ax_plus.imshow(cs_plus_avg, aspect='auto', cmap='jet', 
                                norm=norm, interpolation='nearest')
        
        # CS- heatmap  
        im_minus = ax_minus.imshow(cs_minus_avg, aspect='auto', cmap='jet',
                                  norm=norm, interpolation='nearest')
        
        # Set up axes
        for ax, im in [(ax_plus, im_plus), (ax_minus, im_minus)]:
            # Set time axis ticks and labels
            time_ticks = np.arange(0, len(time_axis), 30)  # Every 30 frames (1 second)
            time_labels = [f'{time_axis[i]:.0f}' for i in time_ticks]
            ax.set_xticks(time_ticks)
            ax.set_xticklabels(time_labels)
            ax.set_xlabel('Time (s)')
            
            # Set ROI axis labels and ticks
            ax.set_yticks(range(0, n_rois, max(1, n_rois//10)))  # Show ROI ticks every 10% of ROIs
            ax.set_ylabel('ROI')
            
            # Set limits to match plot_individual_trial_traces
            time_start_idx = np.argmin(np.abs(time_axis + 1.0))  # -1.0 seconds
            time_end_idx = np.argmin(np.abs(time_axis - 4.0))    # +4.0 seconds
            ax.set_xlim(time_start_idx, time_end_idx)
            ax.set_ylim(-0.5, n_rois - 0.5)  # Center ROIs on pixel centers
            
        # Find stimulus timing indices for horizontal bar positioning
        onset_idx = np.argmin(np.abs(time_axis))  # t=0
        offset_idx = np.argmin(np.abs(time_axis - 2.0))  # t=2s (typical stimulus duration)
        
        # Add session label on the left side of CS+ subplot1 with more spacing
        ax_plus.text(-0.35, 0.5, f'{session_label}', transform=ax_plus.transAxes, 
                    fontsize=12, fontweight='bold', rotation=90, 
                    verticalalignment='center', horizontalalignment='center')
    
    # Add stimulus timing bars and arrows
    for plot_idx, session_label in enumerate(session_labels):
        cached_data = session_data_cache[session_label]
        if cached_data is None:
            continue
            
        time_axis = cached_data['time_axis']
        
        # Get axes for this row
        ax_plus = axes[plot_idx, 0]  # CS+ (left)
        ax_minus = axes[plot_idx, 1]  # CS- (right)
        
        # Find stimulus timing indices
        onset_idx = np.argmin(np.abs(time_axis))  # t=0
        offset_idx = np.argmin(np.abs(time_axis - 2.0))  # t=2s (typical stimulus duration)
        
        # Add horizontal stimulus bars above each plot (outside the plot area)
        for ax, is_cs_plus in [(ax_plus, True), (ax_minus, False)]:
            # Convert time indices to relative positions (0-1) within the plot
            x_start = (onset_idx - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
            x_end = (offset_idx - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
            
            # Position bar above the plot using transform
            ax.plot([x_start, x_end], [1.05, 1.05], 
                   color='black', linewidth=4, solid_capstyle='butt',
                   transform=ax.transAxes, clip_on=False)
            
            # Add downward arrow at stimulus offset (only for CS+)
            if is_cs_plus:
                # Use a simpler approach with plot instead of annotate
                ax.plot([x_end, x_end], [1.20, 1.10], 
                       color='black', linewidth=2, transform=ax.transAxes, clip_on=False)
                # Add arrowhead
                ax.plot([x_end-0.01, x_end, x_end+0.01], [1.11, 1.10, 1.11], 
                       color='black', linewidth=2, transform=ax.transAxes, clip_on=False)
    
    # Add more space between title and plots to prevent overlap
    plt.tight_layout()
    
    # Add a colorbar positioned to the right of the first row only
    if 'im_plus' in locals():
        # Create space for colorbar and title by adjusting subplot positions
        # Add more space at top for title, between rows, and left for labels
        plt.subplots_adjust(right=0.85, top=0.88, hspace=0.5, left=0.20)  # More space for title, between rows, and left for labels
        
        # Position colorbar next to the first row only (same height as one subplot)
        # Calculate the position based on the first row's position
        first_row_bottom = axes[0, 0].get_position().y0
        first_row_top = axes[0, 0].get_position().y1
        first_row_height = first_row_top - first_row_bottom
        
        # Add colorbar to the right of the first row
        cbar_ax = fig.add_axes([0.87, first_row_bottom, 0.03, first_row_height])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_plus, cax=cbar_ax, norm=global_norm)
        cbar.set_label('dF/F', rotation=270, labelpad=15, fontsize=10)
    else:
        # If no colorbar, still add space for the title, between rows, and left for labels
        plt.subplots_adjust(top=0.88, hspace=0.5, left=0.20)  # Add more space for title, between rows, and left for labels
    
    # Save the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        filename = f'trial_averaged_heatmap_{animal_name}.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {filepath}")
    
    plt.show()


def analyze_all_animals(all_data, save_path=None):
    """
    Create trial-averaged heatmaps for all animals in the dataset.
    
    Args:
        all_data (dict): Loaded animal data dictionary
        save_path (str, optional): Path to save heatmaps
    """
    
    # Process each animal
    for animal_name, animal_data in all_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {animal_name}")
        print(f"{'='*60}")
        
        try:
            create_trial_averaged_heatmap(animal_data, animal_name, save_path)
        except Exception as e:
            print(f"Error processing {animal_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Completed processing all animals")
    print(f"{'='*60}")


def analyze_single_animal(all_data, animal_name, save_path=None):
    """
    Create trial-averaged heatmap for a single animal.
    
    Args:
        all_data (dict): Loaded animal data dictionary
        animal_name (str): Name of the animal to analyze
        save_path (str, optional): Path to save heatmap
    """
    
    if animal_name not in all_data:
        available_animals = list(all_data.keys())
        print(f"Animal '{animal_name}' not found. Available animals: {available_animals}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {animal_name}")
    print(f"{'='*60}")
    
    animal_data = all_data[animal_name]
    create_trial_averaged_heatmap(animal_data, animal_name, save_path)


if __name__ == "__main__":
    # Get paths from centralized config
    data_path = processed_data_path
    save_path = os.path.join(save_figures_path, 'heatmaps', 'trial_averaged_heatmaps')
    
    # Load data to show available animals
    print("=" * 60)
    print("TRIAL-AVERAGED HEATMAP GENERATOR")
    print("=" * 60)
    
    try:
        all_data = load_processed_data(data_path)
        available_animals = list(all_data.keys())
        
        print(f"\nAvailable animals: {available_animals}")
        
        print("\nChoose analysis mode:")
        print("1. Single animal analysis")
        print("2. All animals analysis")
        
        while True:
            try:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice in ['1', '2']:
                    break
                else:
                    print("Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nExiting...")
                exit()
        
        if choice == '1':
            # Single animal analysis
            print("\nAvailable animals:")
            for i, animal in enumerate(available_animals, 1):
                print(f"{i}. {animal}")
            
            while True:
                try:
                    animal_choice = input(f"\nSelect animal (1-{len(available_animals)}): ").strip()
                    animal_idx = int(animal_choice) - 1
                    if 0 <= animal_idx < len(available_animals):
                        selected_animal = available_animals[animal_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(available_animals)}")
                except (ValueError, KeyboardInterrupt):
                    print("Please enter a valid number or Ctrl+C to exit")
            
            analyze_single_animal(all_data, selected_animal, save_path)
            
        else:
            # All animals analysis
            print("\nThis will process all animals. Continue? (y/n): ", end="")
            confirm = input().strip().lower()
            if confirm in ['y', 'yes']:
                analyze_all_animals(all_data, save_path)
            else:
                print("Analysis cancelled.")
                
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"Error: {e}")
