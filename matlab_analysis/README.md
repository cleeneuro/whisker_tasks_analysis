# MATLAB Behavioural Analysis

This directory contains MATLAB scripts for behavioral data analysis using the .txt file outputs from the automated behavioural task found here: https://github.com/clee162/whisker_tasks

#### _For readability, its best to open this readme with markdown preview_

## Overview

MATLAB analysis of lick behavior during Pavlovian task. Designed for PavlovianTexture.ino

## Directory Structure

- `lick_analysis/` - Scripts for analyzing lick behavior data
        - `textfile_lickanalysis.m` - Main script for running lick analysis 
        - `lick_rate_plot.m` - Average lick rate plotting utilities
        - `number_licks_histogram.m` - Histogram analysis for number of anticipatory licks 
        - `lick_rate_plot_mouse_all_days.m` - Lick rate plotting for mice across all days
        - `lickrasterplot.m` - Raster plot generation for lick data
        - `import_text_file.m` - Text file import utilities
        - `shadedErrorBar_cl.m` - Shaded error bar plotting (customized colours)
        - `shadedErrorBar.m` - Shaded error bar plotting utilities
- `read_aux_triggers.m` - Auxiliary trigger reading script

## Scripts

### Main Script for running analysis
- **textfile_lickanalysis.m** - Primary script for analyzing lick data from text files

### Functions for analysis/visualization
- **lick_rate_plot.m** - Plots average lick rate across mice
- **number_licks_histogram.m** - Creates histograms for number of licks analysis pooled across mice
- **lick_rate_plot_mouse_all_days.m** - Plots lick rates for individual mice across all experimental days
- **lickrasterplot.m** - Generates raster plots for lick timing data

### Utility scripts 
- **import_text_file.m** - Utility for importing text files containing lick data
- **shadedErrorBar_cl.m** - Adjusted the colours of shaded error bar plotting
- **shadedErrorBar.m** - Standard shaded error bar plotting utilities

### Read auxillary inputs from tiff files
- **read_aux_triggers.m** - Reads and processes auxiliary trigger data

## Usage

### Data structure 

1. Data must be structured as follows. Day indices and whisker trims are detected by counting the # of .txt. files in the directory with the specified naming convention

It is also critical to maintain the textfile formating as outputted from PavlovianTexture.ino
 
   ```
 └── /path/to/your/textfile_data/ 
       └── animalname/
            └── animalname_20240101.txt 
            └── animalname_20240102.txt
            └── .....
            └── animalname_20240110.txt
            └── animalname_20240101_whiskertrim.txt (optional)
   ```

1. All functions are run from `textfile_lickanalysis.m`. User must set:
        - SAVE_FIGURES = true or false 
        - BASE_SAVE_PATH - where you want to save figures
        - outerpath - this is your textfile_data parent folder. Inside this folder, should be animal folders 
        - mousenames - a cell array where user specifies which mice they want to include in the analysis. example mousenames = {'CL019', 'CL020', 'CL021', 'CL022', 'CL023'};
2. Comment/uncomment the functions at the bottom to run specific analyses

# Python version 
- python_analysis/behavioral_analysis/analyze_textpav_textfiles.py contains implementations of similar functionality plus some plots that I never implemented in matlab
        - reads textfiles and automatically finds day indices and whisker trims 
        - Lick rate plots for individual mice 
        - Lick probability 
        - lick index

## read_aux_triggers.m 

- This provides base text for how to read auxillary triggers using matlab
- it produces a basic subplot showing all auxillary triggers 
    - I typically use this for short plots when setting up for experiments. This will be slow on full experiments 
- The python implementation is much better developed. See the python_analysis/README

## Citations 

- **shadedErrorBar.m** - Rob Campbell (2025). raacampbell/shadedErrorBar (https://github.com/raacampbell/shadedErrorBar), GitHub. Retrieved October 20, 2025.