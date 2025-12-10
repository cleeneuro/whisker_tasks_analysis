"""
Learning Phase Mapping 

These functions convert training day labels (Day1, Day2, etc.) 
to learning phases (Early Learning, Late Learning) without modifying the original data.

Usage:
    from learning_phase_mapping import convert_session_labels_to_phases
    
    # Convert labels for an animal's data
    phase_labels = convert_session_labels_to_phases(animal_data)
"""

import re


def day_label_to_learning_phase(day_label):
    """
    Convert a single day label to its corresponding learning phase.
    
    Args:
        day_label (str): Day label like "Day1", "Day2", "Day10", etc.
        
    Returns:
        str: Learning phase ("Early Learning", "Late Learning") or original label if not a day
    """
    # Handle special cases first
    if day_label == "Pre":
        return "Pre"
    elif day_label == "Post":
        return "Post"
    
    # Extract day number from label (e.g., Day1 -> 1, Day10 -> 10)
    match = re.search(r'Day(\d+)', day_label)
    if match:
        day_num = int(match.group(1))
        if day_num <= 2:  # Days 1-2 are Early Learning
            return "Early Learning"
        else:  # Days 3+ are Late Learning
            return "Late Learning"
    else:
        # Not a day label, return as-is (for any other labels)
        return day_label


def convert_session_labels_to_phases(animal_data):
    """
    Convert all session labels in animal data to learning phases.
    Handles missing days by only mapping sessions that actually exist.
    
    Args:
        animal_data (dict): Animal data dictionary containing 'sessions'
        
    Returns:
        dict: Mapping of learning phase labels to session indices
              Format: {phase_label: session_idx}
              
    Example:
        Input sessions: {0: {'label': 'Pre'}, 1: {'label': 'Day2'}, 2: {'label': 'Post'}}
        Output: {'Pre': 0, 'Early Learning': 1, 'Post': 2}
    """
    phase_mapping = {}
    
    # Track multiple sessions per phase (e.g., Day1 and Day2 both become Early Learning)
    phase_sessions = {}
    
    for session_idx, session_data in animal_data['sessions'].items():
        original_label = session_data['label']
        phase_label = day_label_to_learning_phase(original_label)
        
        # Store all sessions for each phase
        if phase_label not in phase_sessions:
            phase_sessions[phase_label] = []
        phase_sessions[phase_label].append(session_idx)
    
    # For each phase, use the earliest (lowest index) session
    # This ensures we only map phases that actually have sessions
    for phase_label, session_indices in phase_sessions.items():
        phase_mapping[phase_label] = min(session_indices)
    
    return phase_mapping


def get_ordered_phase_labels(phase_mapping):
    """
    Get learning phase labels in the correct chronological order.
    
    Args:
        phase_mapping (dict): Mapping of phase labels to session indices
        
    Returns:
        list: Ordered list of phase labels
    """
    preferred_order = ['Pre', 'Early Learning', 'Late Learning', 'Post']
    ordered_labels = []
    
    # Add phases in preferred order if they exist
    for preferred_label in preferred_order:
        if preferred_label in phase_mapping:
            ordered_labels.append(preferred_label)
    
    # Add any remaining phases not in preferred order
    for phase_label in phase_mapping.keys():
        if phase_label not in ordered_labels:
            ordered_labels.append(phase_label)
    
    return ordered_labels


def convert_animal_data_to_phases(animal_data):
    """
    Convenience function that converts animal data and returns ordered phase information.
    
    Args:
        animal_data (dict): Animal data dictionary
        
    Returns:
        tuple: (phase_mapping, ordered_labels)
            - phase_mapping: dict mapping phase labels to session indices
            - ordered_labels: list of phase labels in chronological order
    """
    phase_mapping = convert_session_labels_to_phases(animal_data)
    ordered_labels = get_ordered_phase_labels(phase_mapping)
    
    return phase_mapping, ordered_labels


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    test_cases = [
        "Pre",
        "Day1", 
        "Day2",
        "Day3",
        "Day10",
        "Post",
        "Unknown"
    ]
    
    print("Testing day label conversion:")
    for label in test_cases:
        converted = day_label_to_learning_phase(label)
        print(f"  {label} -> {converted}")
    
    # Test with mock animal data
    mock_animal_data = {
        'sessions': {
            0: {'label': 'Pre'},
            1: {'label': 'Day1'}, 
            2: {'label': 'Day2'},
            3: {'label': 'Day5'},
            4: {'label': 'Post'}
        }
    }
    
    print("\nTesting with mock animal data:")
    phase_mapping, ordered_labels = convert_animal_data_to_phases(mock_animal_data)
    print(f"  Phase mapping: {phase_mapping}")
    print(f"  Ordered labels: {ordered_labels}")
