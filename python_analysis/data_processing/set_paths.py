"""
Simple Path Configuration

Just the basic paths your scripts need. Edit the paths below, and all scripts will use them.
"""

import os

# =============================================================================
# EDIT THESE PATHS - CHANGE ONCE, AFFECTS ALL SCRIPTS
# =============================================================================

#For load_dff_aux_batch.py and similar data processing scripts
tif_root_path = "/path/to/your/imaging/data"
components_root_path = "/path/to/your/caiman_outputs"
save_processed_data_path = "/path/to/your/analysis/processed_data/"

# # For calcium imaging analysis scripts
# processed_data_path = "/path/to/your/analysis/processed_data/auto_discovered_data.pkl"
# save_figures_path = "/path/to/save/analysis/figures"

# For calcium imaging analysis scripts
processed_data_path = "/path/to/your/analysis/processed_data/auto_discovered_data.pkl"
save_figures_path = "/path/to/your/analysis/figures/"

# Create directories if they don't exist
if save_figures_path:
    os.makedirs(save_figures_path, exist_ok=True)
#if save_processed_data_path:
#    os.makedirs(save_processed_data_path, exist_ok=True)

if __name__ == "__main__":
    print("Current path configuration:")
    print(f"tif_root_path: {tif_root_path}")
    print(f"components_root_path: {components_root_path}")
    print(f"save_processed_data_path: {save_processed_data_path}")
    print(f"processed_data_path: {processed_data_path}")
    print(f"save_figures_path: {save_figures_path}")
