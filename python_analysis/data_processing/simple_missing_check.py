"""
Simple script that checks tiff files and caiman outputs 
Looks for sessions with merged dff and tiff files that are missing from the pickle file
without re-loading all the data

This script does not change the existing pickle 
"""

import os
import pickle
import glob

def check_missing_sessions(pickle_path, tif_root_path, components_root_path):
    """Check what sessions are missing without loading all data"""
    
    with open(pickle_path, 'rb') as f:
        existing_data = pickle.load(f)
    
    
    missing_sessions = []
    
    # Scan file system
    for animal in os.listdir(tif_root_path):
        if not os.path.isdir(os.path.join(tif_root_path, animal)):
            continue
            
        animal_tif_path = os.path.join(tif_root_path, animal)
        
        for date_folder in os.listdir(animal_tif_path):
            if not os.path.isdir(os.path.join(animal_tif_path, date_folder)):
                continue
                
            
            # Check if this session exists in pickle file
            session_exists = False
            if animal in existing_data:
                for session_idx, session_data in existing_data[animal]['sessions'].items():
                    existing_date = session_data.get('date')
                    if existing_date == date_folder:
                        session_exists = True
                        break
            
            if not session_exists:
                # Check if required files exist
                tif_files = glob.glob(os.path.join(animal_tif_path, date_folder, f"{animal}_{date_folder}_*.tif"))
                if tif_files:
                    
                    # Check for dF/F file
                    components_path = os.path.join(components_root_path, animal, date_folder)
                    if os.path.exists(components_path):
                        # Look for components folder
                        components_folders = [f for f in os.listdir(components_path) if f.startswith('components_')]
                        if components_folders:
                            dff_path = os.path.join(components_path, components_folders[0], "MergeDuplicates_results", "dff_components.csv")
                            if os.path.exists(dff_path):
                                missing_sessions.append({
                                    'animal': animal,
                                    'date': date_folder,
                                    'tif_path': tif_files[0],
                                    'dff_path': dff_path
                                })
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
    
    return missing_sessions

def main():
    # Import paths from set_paths.py instead of hardcoding
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from set_paths import tif_root_path, components_root_path, processed_data_path
    
    pickle_path = processed_data_path
    
    missing_sessions = check_missing_sessions(pickle_path, tif_root_path, components_root_path)
    
    if not missing_sessions:
        pass
    else:
        print(f"\nFound {len(missing_sessions)} missing sessions:")
        for i, session in enumerate(missing_sessions, 1):
            print(f"  {i}. {session['animal']} - {session['date']}")
            print(f"     TIFF: {session['tif_path']}")
            print(f"     dF/F: {session['dff_path']}")
            print()

if __name__ == "__main__":
    main()
