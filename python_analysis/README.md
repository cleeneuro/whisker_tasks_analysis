## Decoding stimulus-driven responses: Calcium Imaging Analysis Pipeline

_Tip: use your editor's Markdown preview for best readability._

A concise, end-to-end toolkit for calcium imaging, behaviour analysis and single-neuron and population-level decoding. Designed for two-photon imaging of apical dendritic tufts or indiviudal soma during Pavlovian texture discrimination task (https://github.com/clee162/whisker_tasks). Applicable to any calcium imaging dataset with dF/F in .CSV format (ROIs × frames).

## What’s Included

- **Data processing**: Auto-discovers CaImAn CSV outputs and corresponding TIFFs, extracts behaviour task triggers, matches sessions, optional longitudinal ROI tracking, and basic quality control (e.g., checks for frames modified for houselight contamination).
- **Calcium imaging analysis**: 
      - **Statistical Metrics**:  Calculates stimulus selectivity index, stimulus driven AUC/mean dF/F/peak metrics, baseline-corrected options, non-parametric significance testing for stimulus evoked responses. 
      - **Machine Learning & Decoding**:  SVM to decode stimulus identity from population and single neuron activity. Linear regression encoding model to compare weights of CS+, CS-, licking and reward.
- **Visualization**: Average and individual traces, histograms, scatter plots, and heatmaps for trial-averaged and single-trial responses.
- **Batch workflows**: Multi-animal, multi-session handling with consistent day mapping (pre/training/post).


## Quick Start

1) Install dependencies listed in `requirements.txt` (Python 3.7+).
2) Set paths in `data_processing/set_paths.py`.
3) Ensure data layout matches the structure below.
4) Process data to generate the pickle file.
5) Run analysis/visualization scripts.

### Data layout
```
your_data_directory/
├── caiman_outputs/
│   └── animalname/
│       ├── Longitudinal_matches/
│       │   └── final_processed_matches.csv
│       └── date/
│           └── MergeDuplicates_results/dff_components.csv
└── imaging/
    └── animalname/
        ├── datepassive/animalname_datepassive_*.tif
        └── date/animalname_date_*.tif
```

### Configure paths (`data_processing/set_paths.py`)
```python
# For data processing
tif_root_path = "/path/to/your/imaging"
components_root_path = "/path/to/your/caiman_outputs"
save_processed_data_path = "/path/to/your/analysis/processed_data/"

# For analysis scripts
processed_data_path = "/path/to/your/analysis/processed_data/auto_discovered_data.pkl"
save_figures_path = "/path/to/your/analysis/figures"
```

### Process data
```bash
python data_processing/load_dff_aux_batch_auto.py
```
This discovers matching CaImAn outputs and TIFFs, extracts behavioral triggers (CS+/CS-/lick TTLs), loads longitudinal tracking if present, applies valid frame masks when available, and saves a consolidated pickle.

### Analysis + visualization
Run analysis/visualization scripts in /calcium_imaging

## Repository Structure

```
python_analysis/
├── behavioural_analysis/                 # Parse PavlovianTexture text files, visualize behavior across days
├── data_processing/                      # Data loading and preprocessing
│   ├── load_dff_aux_batch_auto.py        # Main data processing
│   ├── read_aux_triggers_tiff.py         # Extract triggers from TIFF aux channels
│   ├── animal_metadata.csv               # Animal information
│   ├── set_paths.py                      # Paths used by analysis scripts
│   ├── learning_phase_mapping.py         # Learning phase definitions
│   └── simple_missing_check.py       # Diagnostic that checks for tiff files missing from pickle
│
└── calcium_imaging/                      # Analysis and visualization
    ├── data_loader.py                    # function for loading pickle
    ├── extract_trial_dff.py              # Extract df/f during trials
    ├── calculate_selectivity.py       # Function for calculating CS+ vs CS- selectivity 
    ├── stimulus_response_stats.py     # Find statistically significant responses
    ├── selectivity_histogram_pooled.py      # Visualize selectivity as histogram, pooled
    ├── selectivity_histogram_singleanimal.py        # Visualize selectivity as histogram
    ├── scatterplot_cs_responses.py          # CS+ vs CS- responses visualization
    ├── histogram_peak_responses.py       # Show peak response amplitudes for CS+ and CS-
    ├── average_traces.py              # Plot trial-averaged responses for each ROI 
    ├── avg_and_individual_traces_all.py     # Plot trial-averaged responses with individual trial overlay
    ├── avg_and_individual_traces_tracked.py       # Plot tracked ROI responses across days
    ├── heatmap_trial_averaged.py         # Heatmap, averaging across trials 
    ├── heatmap_single_trial.py        # Heatmap, single trial, no averaging
    ├── linear_encoding_model.py        # Fit linear encoding model to dF/F
    ├── svm_population_decoding_analysis.py    # Decode CS+ vs CS- trials from population activity 
    └── svm_single_roi_selectivity.py        # Decode CS+ vs CS- trials from single ROI activity 
```

## Inputs and Outputs

- **Inputs**
  - CaImAn dF/F CSVs (`dff_components.csv`, etc.)
  - calcium imaging TIFFs with auxiliary trigger channels (ScanImage/NI resonant)
  - Optional longitudinal ROI matches (`Longitudinal_matches/final_processed_matches.csv`)
  - Animal metadata (`animal_metadata.csv`)

- **Outputs**
  - Pickle: consolidated per-animal data with sessions, triggers, and tracking
  - Figures: PNG/PDF/SVG
  - Analysis summaries (where applicable)

## Configuration and Parameters

- Paths: set once in `data_processing/set_paths.py` (used by downstream scripts).
- Analysis parameters: temporal windows, thresholds, grouping (learners/cheaters), and figure settings are configurable in the analysis scripts.

## Notes on Session Ordering

- Pre-session: first passive (`*passive.tif`)
- Post-session: last passive
- Training: all sessions between pre and post
- Day indices follow folder-date ordering in the imaging folder; skipped days retain their indices.
- If there are no passive sessions, training sessions will still be loaded normally.

## Citations

CaImAn was used to segment apical dendritic tufts and extract dF/F; this package operates on CaImAn dF/F.

Giovannucci A., Friedrich J., Gunn P., Kalfon J., Koay S.A., Taxidis J., Najafi F., Gauthier J.L., Zhou P., Tank D.W., Chklovskii D.B., Pnevmatikakis E.A. (2019). CaImAn: An open source tool for scalable Calcium Imaging data Analysis. eLife 8:e38173.
