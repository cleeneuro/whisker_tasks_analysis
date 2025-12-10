"""
Written by Candice Lee, Sept.2025 

This script automatically discovers the data structure for all animals and their sessions. 
It will save the post merge df/f, the behavioural data (CS+, CS-, lick times), valid_frames_indices (if it exists), tracked ROI indices (if it exists). 

This works by defining (1) a caiman outputs folder and (2) a tiff folder where all the original tiff files are stored. 
If both a final MergeDuplicates_results/dff_components.csv file and a tiff file with THE SAME DATE IN THE FILE NAME are found, then the session is loaded into the dictionary

It uses the animal_metadata.csv file to get the cheater, intrinsic, learner, and houselight information for each animal.
If houselight is set to auto, valid_frames_indices file is found, then the session is considered to have houselight contamination and the valid frames (unmodified; without houselight correction) will be loaded.

It will automatically index the days based on the number of folders within the tiff path with dates in the name.
Even if no df/f data is found, we still use that folder to count the number of training days 
The first passive session is considered the pre-day. 
The last passive session is considered the post-day. 
The training sessions are considered the days in between. 

------------------------------------------------------------------------------
!!!The structure of your data is critical for this to work!!! 

caiman outputs should remain as the default output from the caiman pipeline.
caiman_outputs/animalname/date/components_*.csv 

Tiff files must be saved as: 
- imaging/animalname/date/animalname_date_00001.tif
- If acquisition number 00001 is not found, it will automatically scan for the actual acquisition number.
IMPORTANT: The animal name anddate in the tiff file must be the same as the date in the caiman outputs file. 
You cannot have multiple acquisitions under that name in the folder. 

Passive sessions must be named as imaging/animalname/date/animalname_date_passive.tif
- there must be only two folders with passive in the name. 
- The first will be considered the pre-day. The last will be considered the post-day.

WHAT YOU NEED TO MODIFY: 
- animal_metadata.csv (add your animal information)
- caiman outputs folder
- tiff folder

------------------------------------------------------------------------------


"""

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
from datetime import datetime
from collections import defaultdict
# from animal_config import get_all_cheater_info, get_all_intrinsic_info, get_all_learner_info, get_all_houselight_info, print_animal_summary

def read_animal_metadata_from_csv(csv_path):
    """
    Read animal metadata from a CSV file.
    
    Expected CSV format:
    animal_id,cheater,intrinsic_imaging_result,learner,houselight,sex,mouse_line,notes
    CL030,cheater,n/a,learner,auto,female,GCaMP6f,ITI houselight
    CL031,cheater,n/a,learner,auto,male,GCaMP6f,ITI houselight
    ...
    
    Args:
        csv_path (str): Path to the CSV file containing animal metadata
    
    Returns:
        tuple: (cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info) dictionaries
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Initialize dictionaries
        cheater_info = {}
        intrinsic_info = {}
        learner_info = {}
        houselight_info = {}
        sex_info = {}
        mouse_line_info = {}
        
        # Fill dictionaries from CSV data
        for _, row in df.iterrows():
            animal_id = str(row['animal_id']).strip()
            cheater_info[animal_id] = str(row['cheater']).strip()
            intrinsic_info[animal_id] = str(row['intrinsic_imaging_result']).strip()
            learner_info[animal_id] = str(row['learner']).strip()
            houselight_info[animal_id] = str(row['houselight']).strip()
            sex_info[animal_id] = str(row['sex']).strip()
            mouse_line_info[animal_id] = str(row['mouse_line']).strip()
        
        # print(f"Successfully loaded metadata for {len(cheater_info)} animals from {csv_path}")
        return cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info
        
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        print("Please ensure the CSV file has the following columns:")
        print("animal_id,cheater,intrinsic_imaging_result,learner,houselight,sex,mouse_line,notes")
        return {}, {}, {}, {}, {}, {}

def print_animal_summary_from_csv(cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info):
    """Print a summary of all animals from CSV data."""
    print("Animal Configuration Summary:")
    print("=" * 80)
    
    all_animals = set(cheater_info.keys()) | set(intrinsic_info.keys()) | set(learner_info.keys()) | set(houselight_info.keys())
    
    for animal in sorted(all_animals):
        learner_status = learner_info.get(animal, 'unknown')
        cheater_status = cheater_info.get(animal, 'unknown')
        intrinsic_result = intrinsic_info.get(animal, 'unknown')
        houselight_status = houselight_info.get(animal, 'auto')
        sex = sex_info.get(animal, 'unknown')
        mouse_line = mouse_line_info.get(animal, 'unknown')
        print(f"{animal}: cheater={cheater_status}, intrinsic={intrinsic_result}, learner={learner_status}, houselight={houselight_status}, sex={sex}, mouse_line={mouse_line}")
    
    print(f"\nTotal animals configured: {len(all_animals)}")

def detect_and_load_valid_frames_indices(tif_path):
    """
    If imaging files have been modified due to houselight, there will be a valid_frames_indices file in the same directory as the TIFF file.
    This function detects and loads valid_frames_indices file if it exists in the same directory as the TIFF file.
    
    Args:
        tif_path (str): Path to the TIFF file
        
    Returns:
        tuple: (has_houselight: bool, valid_frames: np.ndarray or None)
    """
    tif_dir = os.path.dirname(tif_path)
    
    # Check for both .csv and .npy files
    csv_path = os.path.join(tif_dir, "valid_frames_indices.csv")
    npy_path = os.path.join(tif_dir, "valid_frames_indices.npy")
    
    if os.path.exists(npy_path):
        try:
            valid_frames = np.load(npy_path)
            # print(f"Loaded valid_frames_indices from {npy_path}")
            return True, valid_frames
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return False, None
            
    elif os.path.exists(csv_path):
        try:
            valid_frames = np.loadtxt(csv_path, delimiter=',', dtype=int)
            # print(f"Loaded valid_frames_indices from {csv_path}")
            return True, valid_frames
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return False, None
    else:
        return False, None

def discover_animal_data_structure(base_tif_path, base_components_path, cheater_info=None, intrinsic_info=None, learner_info=None, houselight_info=None, sex_info=None, mouse_line_info=None):
    """
    Automatically discovers the data structure for all animals and their sessions.
    
    INPUT:
        base_tif_path - Base path to the TIFF files
        base_components_path - Base path to the fluorescence data CSV files
        cheater_info - Dictionary mapping animal names to "cheater" or "non-cheater"
        intrinsic_info - Dictionary mapping animal names to intrinsic imaging results
        learner_info - Dictionary mapping animal names to learner status
        houselight_info - Dictionary mapping animal names to houselight status ("auto", True, False)
    
    OUTPUT:
        data_structure - Dictionary containing discovered animal and session information
    """
    data_structure = {}
    
    # Get only the animals specified in the configuration
    configured_animals = set(cheater_info.keys()) | set(intrinsic_info.keys()) | set(learner_info.keys()) | set(houselight_info.keys())
    # print(f"Processing only configured animals: {sorted(configured_animals)}")
    
    for animal in configured_animals:
        animal_tif_path = os.path.join(base_tif_path, animal)
        animal_components_path = os.path.join(base_components_path, animal)
        
        # Check if animal folder exists in TIFF directory
        if not os.path.exists(animal_tif_path):
            print(f"Warning: No TIFF folder found for {animal}, skipping...")
            continue
        
        # Check if components path exists
        has_components = os.path.exists(animal_components_path)
        if not has_components:
            print(f"Warning: No components folder found for {animal}, skipping...")
            continue
        
        # Get all date folders for this animal (only those starting with date pattern)
        all_date_folders = [d for d in os.listdir(animal_tif_path) 
                           if os.path.isdir(os.path.join(animal_tif_path, d)) and re.match(r'^\d{8}', d)]
        
        if not all_date_folders:
            print(f"Warning: No date folders found for {animal}, skipping...")
            continue
        
        # Filter to only include dates that have dF/F data
        dates_with_dff_data = []
        for date in all_date_folders:
            components_path = find_components_path(base_components_path, animal, date)
            if components_path:
                dff_csv_path = f"{components_path}/MergeDuplicates_results/dff_components.csv"
                if os.path.exists(dff_csv_path):
                    dates_with_dff_data.append(date)
                # else:
                #     print(f"Skipping {animal} {date}: No dF/F data found")
            # else:
            #     print(f"Skipping {animal} {date}: No components found")
        
        if not dates_with_dff_data:
            print(f"Warning: No dates with dF/F data found for {animal}, skipping...")
            continue
        
        # Sort dates chronologically
        dates_with_dff_data.sort()
        # print(f"Found {len(dates_with_dff_data)} dates with dF/F data for {animal}")
        
        # Initialize animal data
        animal_data = {}
        
        # Add cheater information if provided
        if cheater_info and animal in cheater_info:
            animal_data['cheater'] = cheater_info[animal]
        else:
            animal_data['cheater'] = 'unknown'  # Default if not specified
            
        # Add intrinsic imaging result if provided
        if intrinsic_info and animal in intrinsic_info:
            animal_data['intrinsic_imaging_result'] = intrinsic_info[animal]
        else:
            animal_data['intrinsic_imaging_result'] = 'unknown'  # Default if not specified
            
        # Add learner information if provided
        if learner_info and animal in learner_info:
            animal_data['learner'] = learner_info[animal]
        else:
            animal_data['learner'] = 'unknown'  # Default if not specified
            
        # Add houselight information if provided
        if houselight_info and animal in houselight_info:
            animal_data['houselight'] = houselight_info[animal]
        else:
            animal_data['houselight'] = 'auto'  # Default if not specified
            
        # Add sex information if provided
        if sex_info and animal in sex_info:
            animal_data['sex'] = sex_info[animal]
        else:
            animal_data['sex'] = 'unknown'  # Default if not specified
            
        # Add mouse line information if provided
        if mouse_line_info and animal in mouse_line_info:
            animal_data['mouse_line'] = mouse_line_info[animal]
        else:
            animal_data['mouse_line'] = 'unknown'  # Default if not specified
        
        # Categorize sessions based on date patterns
        pre_sessions = []
        post_sessions = []
        training_sessions = []
        
        # Separate passive and training sessions from ALL date folders (not just those with dF/F data)
        all_passive_sessions = [date for date in all_date_folders if 'passive' in date.lower()]
        all_training_sessions = [date for date in all_date_folders if 'passive' not in date.lower()]
        
        # print(f"All training date folders for {animal}: {all_training_sessions}")
        
        # Sort all training sessions chronologically (including those without dF/F data)
        all_training_sessions.sort()
        
        # Separate passive and training sessions from dates WITH dF/F data
        passive_sessions = [date for date in dates_with_dff_data if 'passive' in date.lower()]
        training_sessions = [date for date in dates_with_dff_data if 'passive' not in date.lower()]
        
        # Sort all sessions chronologically to determine pre/post
        all_sessions = dates_with_dff_data.copy()
        all_sessions.sort()
        
        # Sort training sessions chronologically
        training_sessions.sort()
        
        # Determine if passive session is pre or post based on its position relative to training days
        if len(passive_sessions) == 1:
            passive_date = passive_sessions[0]
            
            if len(training_sessions) == 0:
                # No training days, so passive is pre
                pre_sessions.append(passive_date)
            else:
                # Compare passive date with training dates
                if passive_date < min(training_sessions):
                    # Passive is earlier than all training days → Pre
                    pre_sessions.append(passive_date)
                elif passive_date > max(training_sessions):
                    # Passive is later than all training days → Post
                    post_sessions.append(passive_date)
                else:
                    # Passive is between training days → Pre (you can change this logic if needed)
                    pre_sessions.append(passive_date)
                    
        elif len(passive_sessions) > 1:
            # Multiple passive sessions - treat first as pre, last as post
            passive_sessions.sort()
            pre_sessions.append(passive_sessions[0])
            post_sessions.append(passive_sessions[-1])
            
            # Middle passive sessions as additional pre (you can modify this logic if needed)
            if len(passive_sessions) > 2:
                for passive_date in passive_sessions[1:-1]:
                    pre_sessions.append(passive_date)
        
        # Build the session dictionary with chronological indexing
        # Start with -1 for the first pre-day, then count up chronologically
        all_sessions_chronological = sorted(dates_with_dff_data)
        
        # Find the first pre-day to start indexing from -1
        first_pre_idx = -1
        session_idx = first_pre_idx
        
        for date in all_sessions_chronological:
            # Determine label based on session type
            if date in pre_sessions:
                label = "Pre"
            elif date in post_sessions:
                label = "Post"
            elif date in training_sessions:
                # Count training days from 1 based on position in ALL training sessions (including those without dF/F data)
                training_day_num = all_training_sessions.index(date) + 1
                label = f"Day{training_day_num}"
            else:
                label = "Unknown"
            
            # Check for houselight (valid_frames_indices) if houselight is set to "auto"
            session_houselight = False
            if animal_data.get('houselight') == 'auto':
                # Try 00001 first, then detect if needed
                tifpath_00001 = f"{animal_tif_path}/{date}/{animal}_{date}_00001.tif"
                if os.path.exists(tifpath_00001):
                    tifpath = tifpath_00001
                else:
                    detected_acq = detect_acquisition_number(tifpath_00001)
                    tifpath = f"{animal_tif_path}/{date}/{animal}_{date}_{detected_acq}.tif"
                has_houselight, _ = detect_and_load_valid_frames_indices(tifpath)
                session_houselight = has_houselight
            elif animal_data.get('houselight') is True:
                session_houselight = True
            
            # Try 00001 first (common case), then detect if needed
            tifpath_00001 = f"{animal_tif_path}/{date}/{animal}_{date}_00001.tif"
            if os.path.exists(tifpath_00001):
                detected_acq = "00001"
            else:
                # Only scan directory if 00001 doesn't exist
                detected_acq = detect_acquisition_number(tifpath_00001)
                # if detected_acq != "00001":
                #     print(f"Updated acquisition number for {animal} {date}: {detected_acq}")
            
            animal_data[date] = {
                "acq_num": detected_acq,  # Use detected acquisition number
                "label": label,
                "session_index": session_idx,
                "houselight": session_houselight
            }
            session_idx += 1
        
        if animal_data:
            data_structure[animal] = animal_data
            cheater_status = animal_data.get('cheater', 'unknown')
            intrinsic_result = animal_data.get('intrinsic_imaging_result', 'unknown')
            learner_status = animal_data.get('learner', 'unknown')
            houselight_status = animal_data.get('houselight', 'auto')
            sex = animal_data.get('sex', 'unknown')
            mouse_line = animal_data.get('mouse_line', 'unknown')
            session_count = len([k for k in animal_data.keys() if k not in ['cheater', 'intrinsic_imaging_result', 'learner', 'houselight', 'sex', 'mouse_line']])
            components_status = "with components" if has_components else "TIFF only"
            # print(f"Discovered {animal} (cheater: {cheater_status}, intrinsic: {intrinsic_result}, learner: {learner_status}, houselight: {houselight_status}, sex: {sex}, mouse_line: {mouse_line}, {components_status}): {session_count} sessions")
            # for date, details in animal_data.items():
            #     if date not in ['cheater', 'intrinsic_imaging_result', 'learner', 'houselight'] and isinstance(details, dict):
            #         houselight_info = f" (houselight: {details.get('houselight', False)})" if 'houselight' in details else ""
            #         print(f"  {date}: {details['label']}{houselight_info}")
    
    return data_structure

def find_components_path(base_components_path, animal, date):
    """
    Find components path for an animal and date, looking in main folder and subfolders.
    
    Args:
        base_components_path (str): Base path to components
        animal (str): Animal ID
        date (str): Date string
    
    Returns:
        str or None: Path to components folder if found, None otherwise
    """
    if not os.path.exists(base_components_path):
        return None
    
    # Search patterns to try - only look for components that match the specific date
    search_pattern = [
        f"{base_components_path}/{animal}/{date}/components_*",  # Direct path: animal/date/components_*
    ]
    
    for i, pattern in enumerate(search_pattern):
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def detect_acquisition_number(tifpath):
    """
    Detects the actual acquisition number from the TIFF file path.
    Since its usually 00001, first, assume its 00001,
    If not, scan directory to find actual acquisition number.
    
    INPUT:
        tifpath - Path to the TIFF file (may contain wrong acquisition number)
    
    OUTPUT:
        acq_num - Acquisition number as string
    """
    # First, try the most common case: 00001
    # Replace the acquisition number in the path with 00001 and check if file exists
    dir_path = os.path.dirname(tifpath)
    filename = os.path.basename(tifpath)
    
    # Extract animal and date from filename to construct 00001 path
    match = re.search(r'^(.+)_(\d{8})_\d{5}\.tif$', filename)
    if match:
        animal_date = f"{match.group(1)}_{match.group(2)}"
        tifpath_00001 = os.path.join(dir_path, f"{animal_date}_00001.tif")
        
        if os.path.exists(tifpath_00001):
            return "00001"  # Fast path for common case
    
    # If 00001 doesn't exist, scan directory to find actual acquisition number
    if os.path.exists(dir_path):
        tif_files = glob.glob(os.path.join(dir_path, "*.tif"))
        if tif_files:
            # Extract acquisition numbers from all files and find the highest
            acq_numbers = []
            for tif_file in tif_files:
                file_basename = os.path.basename(tif_file)
                match = re.search(r'_(\d{5})\.tif$', file_basename)
                if match:
                    acq_numbers.append(int(match.group(1)))
            
            if acq_numbers:
                # Return the highest acquisition number (most recent)
                highest_acq = max(acq_numbers)
                # print(f"Detected acquisition number {highest_acq:05d} from {len(acq_numbers)} files in {dir_path}")
                return f"{highest_acq:05d}"
    
    return "00001"  # Default fallback

def LoadSingleDayData(tifpath, dff_csv_path=None, session_houselight=False):
    """
    Extracts pre-day data, including auxiliary trigger frames and optionally fluorescence responses.

    INPUT:
        tifpath - Path to the TIFF file
        dff_csv_path - Path to the fluorescence data CSV file (optional)
        session_houselight - Boolean indicating if this session has houselight (valid_frames_indices)

    OUTPUT:
        session_data - Dictionary containing pre-day data
    """
    # Initialize the session data dictionary
    session_data = {}

    # Optionally load the fluorescence data
    if dff_csv_path is not None and os.path.exists(dff_csv_path):
        dff = np.loadtxt(dff_csv_path, delimiter=',')
        session_data['dff'] = dff
        # print(f"Loaded dF/F data from {dff_csv_path}")
    # else:
    #     print(f"No dF/F data found for {tifpath}")

    # Load valid_frames_indices if houselight is detected
    if session_houselight:
        has_houselight, valid_frames = detect_and_load_valid_frames_indices(tifpath)
        if has_houselight and valid_frames is not None:
            session_data['valid_frames_indices'] = valid_frames
            session_data['houselight'] = True
            # print(f"Loaded valid_frames_indices with {len(valid_frames)} frames")
        else:
            session_data['houselight'] = False
            print(f"Houselight expected but valid_frames_indices not found")
    else:
        session_data['houselight'] = False

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
        print(f"\nLoading animal {animal}")
        animal_data = {}
        animal_data['sessions'] = {}
        
        # Create metadata for the animal
        animal_data['day_sequence'] = []
        animal_data['day_mapping'] = {}
        animal_data['date_mapping'] = {}
        animal_data['session_types'] = {
            'pre_session': None,
            'training_sessions': [],
            'post_session': None
        }

        # Filter out metadata fields and only process session data
        session_dates = {k: v for k, v in dates.items() 
                        if k not in ['cheater', 'intrinsic_imaging_result', 'learner'] and isinstance(v, dict)}
        
        # Sort dates by session index to maintain chronological order
        sorted_dates = sorted(session_dates.items(), key=lambda x: x[1]['session_index'])
        
        # Get the first date for longitudinal tracking
        first_date = sorted_dates[0][0] if sorted_dates else None

        for session_idx, (date, details) in enumerate(sorted_dates):
            # Update acquisition number if needed
            tifpath = f"{base_tif_path}/{animal}/{date}/{animal}_{date}_{details['acq_num']}.tif"
            if not os.path.exists(tifpath):
                # Try to detect the actual acquisition number
                detected_acq = detect_acquisition_number(tifpath)
                # if detected_acq != details['acq_num']:
                #     print(f"Corrected acquisition number for {animal} {date}: {details['acq_num']} -> {detected_acq}")
                details['acq_num'] = detected_acq
                tifpath = f"{base_tif_path}/{animal}/{date}/{animal}_{date}_{details['acq_num']}.tif"

            # Construct file paths for TIFF and CSV
            # We already verified dF/F data exists in the discovery phase
            components_path = find_components_path(base_components_path, animal, date)
            dff_csv_path = f"{components_path}/MergeDuplicates_results/dff_components.csv"
            print(f"  Loading {details['label']} ({date})")
            print(f"    TIFF: {tifpath}")
            print(f"    dF/F: {dff_csv_path}")

            # Load the longitudinally tracked indices
            if date == first_date and long_track:
                # Look for tracked ROIs in the animal-level Longitudinal_matches folder
                animal_components_path = os.path.join(base_components_path, animal)
                tracked_ROIs_path = f"{animal_components_path}/Longitudinal_matches/final_processed_matches.csv"
                if os.path.exists(tracked_ROIs_path):
                    tracked_ROIs = np.genfromtxt(tracked_ROIs_path, delimiter=',', filling_values=np.nan)
                    animal_data['Tracked ROIs'] = tracked_ROIs
                    print(f"    Tracked ROIs: {tracked_ROIs_path}")
                else:
                    print(f"    Warning: No tracked ROIs found at {tracked_ROIs_path}")

            # Load single day data
            session_houselight = details.get('houselight', False)
            session_data = LoadSingleDayData(tifpath, dff_csv_path, session_houselight)

            # Store in animal's data dictionary
            animal_data['sessions'][session_idx] = session_data
            animal_data['sessions'][session_idx]['label'] = details["label"]
            animal_data['sessions'][session_idx]['date'] = date

            # Update metadata
            animal_data['day_sequence'].append(details["label"])
            animal_data['day_mapping'][details["label"]] = session_idx
            animal_data['date_mapping'][date] = details["label"]
            
            # Update session types
            if details["label"] == "Pre":
                animal_data['session_types']['pre_session'] = session_idx
            elif details["label"] == "Post":
                animal_data['session_types']['post_session'] = session_idx
            elif details["label"].startswith("Day"):
                animal_data['session_types']['training_sessions'].append(session_idx)

            # print('Finished loading data for', animal, date, f"({details['label']})")

        # Add all days' data to the main dictionary for the animal
        all_animal_data[animal] = animal_data
        print("====================")

    return all_animal_data

# Import paths from centralized config
from set_paths import tif_root_path, components_root_path, save_processed_data_path

# Load animal metadata from CSV file
csv_path = "animal_metadata.csv"  # Update this path as needed
cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info = read_animal_metadata_from_csv(csv_path)

# Print animal configuration summary
#print("Animal Configuration:")
#print_animal_summary_from_csv(cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info)
#print()

# Automatically discover the data structure
data = discover_animal_data_structure(tif_root_path, components_root_path, cheater_info, intrinsic_info, learner_info, houselight_info, sex_info, mouse_line_info)

if not data:
    print("No data found! Check your paths.")
else:
    # Print discovery summary
    print(f"Discovered {len(data)} animals with dF/F data")
    for animal, animal_data in data.items():
        print(f"{animal}:")
        # Sort sessions by session index to maintain chronological order
        session_dates = {k: v for k, v in animal_data.items() 
                        if k not in ['cheater', 'intrinsic_imaging_result', 'learner', 'houselight', 'sex', 'mouse_line'] and isinstance(v, dict)}
        sorted_sessions = sorted(session_dates.items(), key=lambda x: x[1]['session_index'])
        
        for date, details in sorted_sessions:
            print(f"  {details['label']} - {date}")
    
    # Load and save all animal data
    print("\nLoading data...")
    all_data = LoadAndSaveAnimalData(data, tif_root_path, components_root_path, long_track=True)

    # Save the dictionary to a Pickle file
    output_filename = os.path.join(save_processed_data_path, "auto_discovered_data.pkl")
    with open(output_filename, "wb") as pickle_file:
        pickle.dump(all_data, pickle_file)
    print(f"\nData loading complete. Saved to {output_filename}") 