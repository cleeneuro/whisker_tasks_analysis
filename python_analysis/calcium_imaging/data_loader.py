"""
Data loading functions for calcium imaging analysis.

This module handles loading processed calcium imaging data from pickle files
and provides functions to list available animals and sessions in the dataset.
"""

import pickle

# Global variable to track if we've already shown the data path
_data_path_shown = set()

def load_processed_data(data_path):
    """
    Load the dictionary saved from load_dff_aux_batch_auto.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        
    Returns:
        dict: Loaded animal data dictionary
    """
    global _data_path_shown
    
    # Only show data path once per unique path
    is_first_load = data_path not in _data_path_shown
    if is_first_load:
        print(f"Loading data from {data_path}...")
        _data_path_shown.add(data_path)
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Only show loaded data summary on first load
    if is_first_load:
        print(f"Loaded data for {len(all_data)} animals")
    
    return all_data


def list_available_animals_and_sessions(data_path):
    """
    List all available animals and their sessions in the dataset.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
        
    Returns:
        dict: Dictionary with animal names as keys and lists of session labels as values
    """
    all_data = load_processed_data(data_path)
    
    animal_sessions = {}
    for animal_name, animal_data in all_data.items():
        sessions = animal_data['sessions']
        session_labels = [session_data['label'] for session_data in sessions.values()]
        animal_sessions[animal_name] = session_labels
    
    return animal_sessions


def print_available_data(data_path):
    """
    Print all available animals and sessions in a formatted way.
    
    Args:
        data_path (str): Path to the pickle file containing processed data
    """
    animal_sessions = list_available_animals_and_sessions(data_path)
    
    print("Available animals and sessions:")
    print("=" * 50)
    for animal_name, sessions in animal_sessions.items():
        print(f"{animal_name}:")
        for session in sessions:
            print(f"  - {session}")
        print()

