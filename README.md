# Behaviour Analysis Toolkit

A comprehensive analysis toolkit for calcium imaging and behavioral data from Pavlovian texture discrimination experiments. This repository contains both **Python** and **MATLAB** implementations for analyzing two-photon calcium imaging and licking behavior during associative learning.

Code for the behavioural task can be found at https://github.com/clee162/whisker_tasks.
## Overview

End-to-end pipelines for:
- **Calcium imaging**: Neural decoding, selectivity analysis, population statistics (Python)
- **Behavior**: Lick rate, anticipatory licking, discrimination metrics (Python & MATLAB)
- **Machine learning**: SVM decoding, linear encoding models (Python)
- **Visualization**: Heatmaps, traces, raster plots, histograms

Designed for two-photon imaging during Pavlovian conditioning, but adaptable to any calcium imaging dataset with dF/F in CSV format.

### 🐍 Python Analysis → [`python_analysis/`](python_analysis/)
**Use for:** Calcium imaging analysis, automated data processing, machine learning, batch workflows

**Key features:**
- Automated data discovery and trigger extraction from TIFFs
- Behavioural analysis 
- Longitudinal ROI tracking across sessions
- Statistical metrics: selectivity indices, AUC, significance testing
- SVM population/single-neuron decoding
- Linear encoding models
- Comprehensive visualization suite

**Quick start:**
```bash
pip install -r requirements.txt
cd python_analysis/data_processing
python load_dff_aux_batch_auto.py
# Then run scripts in calcium_imaging/
```

📖 **[Full Python Documentation →](python_analysis/README.md)**

---

### 🧮 MATLAB Analysis → [`matlab_analysis/`](matlab_analysis/)
**Use for:** Lick behavior analysis of text file output from whisker_tasks

**Key features:**
- Lick rate plots across days and cohorts
- Anticipatory lick quantification
- Raster plots with shaded error bars
- Histogram analysis
- Whisker trim session detection

**Quick start:**
```matlab
% Edit matlab_analysis/lick_analysis/textfile_lickanalysis.m
% Set paths and animal IDs, then run
```

📖 **[Full MATLAB Documentation →](matlab_analysis/README.md)**

---

## Repository Structure

```
BehaviourAnalysis/
├── python_analysis/          # Python pipeline (see dedicated README)
│   ├── calcium_imaging/      # Analysis & visualization scripts
│   ├── data_processing/      # Data loading, trigger extraction
│   └── behavioral_analysis/  # Lick analysis (Python version)
├── matlab_analysis/          # MATLAB pipeline (see dedicated README)
│   └── lick_analysis/        # Lick behavior scripts
├── data/                     # Data directory (not tracked)
│   ├── raw/                  # Raw experimental data
│   └── processed/            # Processed outputs
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

## Installation

### Python
```bash
pip install -r requirements.txt
```
Requires Python 3.7+ with numpy, pandas, scipy, matplotlib, scikit-learn, tifffile.

### MATLAB
- Base MATLAB installation
- Image Processing Toolbox (optional, for TIFFs)
- Third-party: `shadedErrorBar.m` (included)

## Data Formats

- **CaImAn outputs**: `dff_components.csv` (ROIs × frames)
- **Imaging TIFFs**: ScanImage/NI resonant with auxiliary triggers
- **Behavioral text**: Arduino output from PavlovianTexture.ino
- **Tracking**: Longitudinal ROI matches (optional)

## Citations

If you use this code, please cite: 

```
@software{lee2025whiskertasks,
  author = {Lee, Candice},
  title = {Whisker Tasks Analysis: Behavioural and Calcium Imaging Analysis},
  year = {2025},
  url = {https://github.com/cleeneuro/whisker_tasks_analysis},
  version = {1.0.0}
}
```

**CaImAn:**  
Giovannucci A., Friedrich J., et al. (2019). CaImAn: An open source tool for scalable Calcium Imaging data Analysis. *eLife* 8:e38173.

**shadedErrorBar:**  
Rob Campbell (2025). raacampbell/shadedErrorBar. GitHub. https://github.com/raacampbell/shadedErrorBar

## License

MIT License - see [LICENSE](LICENSE) file.

---

**📖 For detailed documentation, see:**
- **[Python Analysis README](python_analysis/README.md)** - Complete guide to calcium imaging pipeline
- **[MATLAB Analysis README](matlab_analysis/README.md)** - Complete guide to lick behavior analysis
