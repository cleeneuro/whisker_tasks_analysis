""" 
Load animal data for a single day and save it to a pickle file.
In most cases, you should use load_dff_aux_batch_auto.py instead.

This version requires more manual inputs. 
Must specify:
- the animal name
- date
- acquisition number.
- the path to the tiff file
- the path to the dff_components.csv file
- the path to the valid_frames_indices.npy file
- trial label (like pre or day 1)

""""

import glob
from read_aux_triggers_tiff import GetAuxTriggers
import re
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
import pandas as pd
import os
from matplotlib.collections import PolyCollection
import pickle

def LoadSingleDayData(tifpath, dff_csv_path=None):
    """
    Extracts pre-day data, including auxiliary trigger frames and optionally fluorescence responses.

    INPUT:
        tifpath - Path to the TIFF file
        dff_csv_path - Path to the fluorescence data CSV file (optional)

    OUTPUT:
        session_data - Dictionary containing pre-day data
    """
    # Initialize the session data dictionary
    session_data = {}

    # Optionally load the fluorescence data
    if dff_csv_path is not None and os.path.exists(dff_csv_path):
        
        dff = np.loadtxt(dff_csv_path, delimiter=',')
        session_data['dff'] = dff
        print(f"Loaded dF/F data from {dff_csv_path}")
    else:
        print(f"No dF/F data found for {tifpath}")

    # Extract auxiliary triggers
    cs_plus_frames, cs_minus_frames, lick_frames = GetAuxTriggers(tifpath)
    session_data.update({
        'cs_plus_frames': cs_plus_frames,
        'cs_minus_frames': cs_minus_frames,
        'lick_frames': lick_frames
    })

    return session_data


def LoadAndSaveAnimalData(data, base_tif_path, base_components_path, long_track=False):
    """
    Loads data for all days of a single animal and combines them.

    INPUT:
        data - Dictionary containing animal and day information
        base_tif_path - Base path to the TIFF files
        base_components_path - Base path to the fluorescence data CSV files

    OUTPUT:
        all_animal_data - Dictionary containing all days' data for the animal
    """
    all_animal_data = {}

    for animal, dates in data.items():
        animal_data = {}
        animal_data['sessions'] = {}

        # Get the first date directly
        first_date = next(iter(dates))

        for date_index, (date, details) in enumerate(dates.items()): 
            acq_num = details["acq_num"]

            # Construct file paths for TIFF and CSV
            tifpath = f"{base_tif_path}/{animal}/{date}/{animal}_{date}_{acq_num}.tif"
            components_path = glob.glob(f"{base_components_path}/{animal}/{date}/components_*")
            if components_path:
                components_path = components_path[0]
            else:
                print(f"No components found for {animal} on {date}")
                continue
            dff_csv_path = f"{components_path}/MergeDuplicates_results/dff_components.csv"

            # Load the longitudinally tracked indices
            if date == first_date and long_track:
                tracked_ROIs_path = f"{components_path}/MergeDuplicates_results/8bitstack/Longitudinal_matches/final_processed_matches.csv"
                tracked_ROIs = np.genfromtxt(tracked_ROIs_path, delimiter=',', filling_values=np.nan)
                # Add to the dictionary
                animal_data['Tracked ROIs'] = tracked_ROIs
                print('Loaded tracked ROIs for', animal, date)

            # Load single day data
            session_data = LoadSingleDayData(tifpath, dff_csv_path)

            # Store in animal's data dictionary
            animal_data['sessions'][date_index] = session_data
            animal_data['sessions'][date_index]['label'] = details["label"]
            animal_data['sessions'][date_index]['date'] = date

            print('Loaded data for', animal, date)

        # Add all days' data to the main dictionary for the animal
        all_animal_data[animal] = animal_data

    return all_animal_data


data = {
    "CL039": {
        "20250322passive": {"acq_num": "00001", "label": "Pre"},
      #  "20240731": {"acq_num": "00001", "label": "Naive"},
       # "20240804": {"acq_num": "00001", "label": "Expert"},
        "20250406passive": {"acq_num": "00001", "label": "Post"}
    },
    "CL040": {
        "20250322passive": {"acq_num": "00001", "label": "Pre"},
        #"20240731": {"acq_num": "00002", "label": "Naive"},
        #"20240804": {"acq_num": "00001", "label": "Expert"},
        "20250406passive": {"acq_num": "00001", "label": "Post"}
    },

}

# Import paths from centralized config
from set_paths import tif_root_path, components_root_path, save_processed_data_path

# Load and save all animal data
all_data = LoadAndSaveAnimalData(data, tif_root_path, components_root_path, long_track=True)

# Save the dictionary to a Pickle file
output_file = os.path.join(save_processed_data_path, "CL039_CL040_data.pkl")
with open(output_file, "wb") as pickle_file:
    pickle.dump(all_data, pickle_file)
print(f'Data saved to {output_file}')