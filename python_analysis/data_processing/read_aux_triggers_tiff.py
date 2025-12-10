import re
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile

def read_aux_triggers(tifpath, save_fig=True):
    """
    Reads the timestamps from aux triggers 0-3 for each frame from the
    ScanImage TIFF metadata and makes a plot.
    
    INPUT:
        tifpath - Path to the ScanImage TIFF stack
    
    OUTPUT:
        aux1, aux2, aux3, metadata - Data arrays and metadata
    """
    # Open the TIFF file and load the metadata
    with TiffFile(tifpath) as tif:
        metadata = [page.description for page in tif.pages]

    # Initialize aux variables as empty lists
    aux0, aux1, aux2, aux3 = [], [], [], []

    # Regular expressions to extract the auxiliary trigger values
    regex_patterns = {
        "aux0": r"auxTrigger0 = \[(.*?)\]",
        "aux1": r"auxTrigger1 = \[(.*?)\]",
        "aux2": r"auxTrigger2 = \[(.*?)\]",
        "aux3": r"auxTrigger3 = \[(.*?)\]"
    }

    for d in metadata:
        # Extract and clean aux triggers, handling commas and empty cases
        aux0_values = re.findall(regex_patterns["aux0"], d)
        aux1_values = re.findall(regex_patterns["aux1"], d)
        aux2_values = re.findall(regex_patterns["aux2"], d)
        aux3_values = re.findall(regex_patterns["aux3"], d)

        aux0.append(list(map(float, aux0_values[0].replace(',', ' ').split())) if aux0_values else [])
        aux1.append(list(map(float, aux1_values[0].replace(',', ' ').split())) if aux1_values else [])
        aux2.append(list(map(float, aux2_values[0].replace(',', ' ').split())) if aux2_values else [])
        aux3.append(list(map(float, aux3_values[0].replace(',', ' ').split())) if aux3_values else [])
    # Identify non-empty cells
    non_empty_cells = {
        "aux0": [bool(vals) for vals in aux0],
        "aux1": [bool(vals) for vals in aux1],
        "aux2": [bool(vals) for vals in aux2],
        "aux3": [bool(vals) for vals in aux3]
    }

    # Convert the Boolean list for aux1 to a NumPy array
    aux0_boolean = np.array(non_empty_cells["aux0"])
    aux1_boolean = np.array(non_empty_cells["aux1"])
    aux2_boolean = np.array(non_empty_cells["aux2"])
    aux3_boolean = np.array(non_empty_cells["aux3"])

    # Plotting
    if save_fig:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        aux_titles = ["Aux0 - air compressor", "Aux1 - CS plus", "Aux2 - CS minus", "Aux3 - lick"]
        for i, (key, ax) in enumerate(zip(non_empty_cells.keys(), axs.flatten())):
            ax.plot(non_empty_cells[key], label=key)
            ax.set_title(aux_titles[i])
            ax.set_ylim([-1, 2])
            ax.set_xlabel('Frames')
            ax.set_ylabel('TTL input')
            ax.legend()

        plt.tight_layout()
        fig_savepath = '/path/to/save/figures/Aux.png'  # Update this path as needed
        plt.savefig(fig_savepath)
        print('Figure saved to:', fig_savepath)

    return aux0_boolean, aux1_boolean, aux2_boolean, aux3_boolean, metadata

def get_stimulus_frames(aux_input_boolean, min_ITI_duration=4, max_stim_duration=3):
    """
    Extracts the start and stop frame numbers for each stimulus and stores them in an array.

    INPUT:
        aux1 - Boolean array (0 or 1) with single 1s marking start and end of stimuli
        min_between_stimuli - Minimum duration (in seconds) between stimuli
        max_stim_duration - Maximum duration (in seconds) of a stimulus
        frame_time - Duration of each frame (in seconds)
    
    OUTPUT:
        stimulus_frames - 2D array where each row is [start_frame, stop_frame]
    """

    # Find all indices where aux1 == 1
    one_indices = np.where(aux_input_boolean == 1)[0]
    
    stimulus_frames = []  # Initialize the array to store start and stop frames
    
    # Iterate through indices and group into start/stop pairs
    for i in range(len(one_indices) - 1):
        start_frame = one_indices[i]
        stop_frame = one_indices[i + 1]
        #next_trial_start = one_indices[i + 2] if i + 2 < len(one_indices) else len(aux_input_boolean)

        # Calculate durations in frames
        stim_duration = (stop_frame - start_frame) * 1/FR
        #iti = (next_trial_start - stop_frame) * 1/FR
        
        # Check timing constraints
        #if stim_duration <= max_stim_duration and iti >= min_ITI_duration:
        if stim_duration <= max_stim_duration:
            stimulus_frames.append([start_frame, stop_frame])
    
    return np.array(stimulus_frames)

def get_lick_frames(aux3_input_boolean):
    """
    Extracts the frame numbers for each lick and stores them in an array

    INPUT:
        aux3 - Boolean array (0 or 1) with single 1s marking licks
        fr - Frame rate (frames per second)
    
    OUTPUT:
        lick_frames - Array of frame numbers for each lick
    """    
    # Find all indices where aux3 == 1
    lick_frames = np.where(aux3_input_boolean == 1)[0]
    
    return lick_frames

def GetAuxTriggers(tifpath):
    print ('Reading aux triggers from:', tifpath)
    aux0_boolean, aux1_boolean, aux2_boolean, aux3_boolean, metadata = read_aux_triggers(tifpath, save_fig=False)
    #print('Getting stimulus frames...')
    cs_plus_frames = get_stimulus_frames(aux1_boolean, min_ITI_duration=4, max_stim_duration=3)
    cs_minus_frames = get_stimulus_frames(aux2_boolean, min_ITI_duration=4, max_stim_duration=3) 
    #print('Getting lick frames...')
    lick_frames = get_lick_frames(aux3_boolean)  
    #print('CS_plus_frames shape:', cs_plus_frames.shape)
    #print('CS_minus_frames shape:', cs_minus_frames.shape)
    #print('Lick_frames shape:', lick_frames.shape)
    
    # if cs_plus frames, cs_minus frames, or lick frames are empty, print a warning
    if cs_plus_frames.shape[0] == 0:
        print('Warning: CS+ frames are empty')
    if cs_minus_frames.shape[0] == 0:
        print('Warning: CS- frames are empty')
    if lick_frames.shape[0] == 0:
        print('Warning: Lick frames are empty')

    print('Finished getting aux triggers for', tifpath)

    return cs_plus_frames, cs_minus_frames, lick_frames
