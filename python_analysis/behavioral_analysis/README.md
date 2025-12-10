# Analyze textfile outputs from whisker_tasks PavlovianTexture 

## Overview 

This script loads, parses and analyzes textfile outputs saved from the automated behavioural task found here: https://github.com/clee162/whisker_tasks

The `analyze_pavlovian_textfiles.py` script processes Pavlovian conditioning data from Arduino-based behavioral tasks (PavlovianTexture or GradedPavlovianTexture) and generates comprehensive visualizations of licking behavior across training sessions.

## What's Included

- **Lick Index Analysis**: Calculates and visualizes lick index (CS+ vs CS- licking) across training days, including whisker trim sessions
- **Anticipatory Licking Analysis**: Measures probability of anticipatory licking (licking before reward delivery) for CS+ and CS- trials
- **Average Lick Rate Over Time**: Time-resolved lick rate analysis showing average licking behavior across all mice before and after CS onset
- **Lick Rate by Mouse**: Individual mouse lick rate trajectories across training days

## Quick Start

1. **Configure paths** in `analyze_pavlovian_textfiles.py`:
   ```python
   DATA_DIRECTORY = '/path/to/your/textfile_data/'
   BASE_OUTPUT_FOLDER = '/path/to/your/output_figures/'
   MOUSE_FOLDERS = ['mousename1', 'mousename2', 'mousename3']  # Or None to include all folders
   ```

2. **Ensure data structure** matches:
   ```
   DATA_DIRECTORY/
   ├── mousename1/
   │   ├── mousename1_file1.txt
   │   ├── mousename1_file2.txt
   │   └── mousename1_file3_whiskertrim.txt  # Optional
   └── mousename2/
       └── ...
   ```

3. **Run the script**:
   ```bash
   python analyze_pavlovian_textfiles.py
   ```

4. **Select analyses** from the interactive menu:
   - Option 1-4: Run individual analyses
   - Option 5: Run all analyses
   - Option 6: Exit

## Data Requirements

- Textfile outputs from PavlovianTexture or GradedPavlovianTexture Arduino code
- Files must be organized in folders named by mouse identifier and in order of date/session #
- Training files: regular `.txt` files (excluding files with '@' in the name)
- Optional whisker trim files: files containing `_whiskertrim.txt` in the filename

## Analysis Details

### Lick Index Analysis
- Calculates lick index: `(CS+ licks - CS- licks) / (CS+ licks + CS- licks)`
- Tracks learning progression across training days
- Separately handles whisker trim sessions if present
- Generates line plots showing lick index evolution

### Anticipatory Licking Analysis
- Measures probability of licking within a threshold time window before reward delivery
- Compares CS+ vs CS- anticipatory licking
- Default threshold: 3 seconds before reward (configurable via `ANT_LICK_THRESHOLD`)

### Lick Rate Over Time
- Bins lick events in time windows (default: 200 ms bins)
- Analyzes 3 seconds before and 8 seconds after CS onset
- Shows average lick rate across all mice and individual mouse trajectories

## Configuration Parameters

Key parameters can be adjusted in the script:

- `SERVO_MOVE_DUR`: Servo movement duration (ms) - default: 1120
- `STIM_PRESENT_DUR`: Stimulus presentation duration (ms) - default: 2000
- `ANT_LICK_THRESHOLD`: Threshold for anticipatory licking (s) - default: 3
- `LICK_RATE_BIN_SIZE`: Time bin size for lick rate analysis (ms) - default: 200
- `LICK_RATE_PRE_TIME`: Time before CS onset to analyze (ms) - default: 3000
- `LICK_RATE_POST_TIME`: Time after CS onset to analyze (ms) - default: 8000
- `REWARD_TIME_OFFSET`: Time when reward is delivered (ms) - default: 2700

## Outputs

All figures are saved to `BASE_OUTPUT_FOLDER` (or a subfolder named after selected mice):
- PNG files with 300 DPI resolution
- Interactive matplotlib windows for immediate viewing
- Organized by analysis type and mouse (where applicable)

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib

## Notes

- The script automatically excludes files with '@' in the filename
- Whisker trim sessions are handled separately in lick index and anticipatory licking analyses
- Output folder structure is automatically created based on selected mice
