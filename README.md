## Consumer Survey Data Analysis using Structural Equation Modelling

This project demonstrates how to analyze survey data using **Structural Equation Modeling (SEM)** and visualize the results using **MATLAB**. The scripts are designed to be adaptable and can be customized to fit any survey data and research objectives.

## Project Overview

This repository provides a comprehensive workflow for conducting SEM analysis on consumer survey data. It includes:

1. **Data Preprocessing:** Handling categorical and numerical data, applying mappings, and normalization.
2. **SEM Analysis using Python:** Setting up and performing SEM analysis with the `semopy` package.
3. **MATLAB Visualization:** Visualizing SEM results, including path diagrams, using MATLAB.

This repository is organized into the following key sections:

- **Data Preprocessing and SEM Analysis in Python**
- **Visualizing SEM Results in MATLAB**
- **Customization Guide** for modifying the scripts to suit different datasets.

## Getting Started

### Prerequisites

1. **Python Libraries**
   - `pandas`
   - `numpy`
   - `semopy`
   - `matplotlib`
   - `seaborn`
   - `scipy`
   - `sklearn`

2. **MATLAB** (for visualizing SEM results)
   - Basic understanding of how to work with `.mat` files in MATLAB.
   - MATLAB installed with any required visualization toolboxes.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/consumer-survey-sem.git
    cd consumer-survey-sem
    ```

2. Install the necessary Python libraries:

    ```bash
    pip install pandas numpy semopy matplotlib seaborn scipy scikit-learn
    ```

3. Make sure MATLAB is set up and configured on your system for visualizing the `.mat` files.

### Folder Structure

```plaintext
consumer-survey-sem/
│
├── data/
│   └── consumer_survey_dataset.csv   # Sample survey dataset (Replace with your own data)
│
├── scripts/
│   ├── sem_analysis.py               # Python script for SEM analysis and exporting data for MATLAB
│   ├── visualize_in_matlab.m         # MATLAB script for visualizing SEM results
│   └── preprocessing_and_visualization.py  # Python script for preprocessing data and basic visualizations
│
├── README.md                         # This readme file
└── results/
    └── sem_results.mat               # Exported SEM results for MATLAB visualization
```

### Data Preparation

1. Place your dataset in the `data/` folder. The dataset should be in `.csv` format.
2. Make sure the dataset is formatted correctly with appropriate column headers and data types.

### How to Use

#### Step 1: Preprocess and Analyze Data using Python

- Open the `scripts/sem_analysis.py` file.
- Modify the **model specification** and **dataset path** based on your survey dataset.
- Run the script to perform SEM analysis and save results for MATLAB visualization:

    ```bash
    python scripts/sem_analysis.py
    ```

- The script will generate SEM results in `.mat` format (MATLAB file) in the `results/` folder.

#### Step 2: Visualize SEM Results using MATLAB

- Open MATLAB and navigate to the repository's `scripts/` folder.
- Open `visualize_in_matlab.m` and adjust the script to match your analysis needs.
- Run the script in MATLAB to visualize SEM results.

```matlab
% Open and run the script in MATLAB
visualize_in_matlab.m
```

This will generate path diagrams and any other visualizations defined in the MATLAB script.

## Customization Guide

### Python Script Customization

1. **Preprocessing & Data Mapping** (`preprocessing_and_visualization.py`)
   - Customize the `mappings` dictionary to match the categories in your dataset.
   - Modify the list of numerical columns to include your specific variables.

2. **SEM Analysis** (`sem_analysis.py`)
   - Update the SEM model specification in `model = """ ... """`.
   - Adjust column names and relationships as needed for your dataset.

### MATLAB Script Customization

1. **Loading Data**:
   - Ensure the `.mat` file in the `results/` folder is correctly referenced.

2. **Visualization Parameters**:
   - Modify path diagrams and visual elements based on your research needs.

The resulting SEM analysis and MATLAB visualizations will provide insights into these relationships, which you can use for academic research, industry reports, or business strategy.

## FAQ

1. **How do I adjust the SEM model?**
   - Edit the `model` string in the `sem_analysis.py` script. Define your variables and relationships according to your theoretical framework.

2. **Can I use this for other survey data?**
   - Yes! Simply replace the dataset and update the mappings, SEM model, and visualizations to fit your needs.

3. **How do I visualize the results in MATLAB?**
   - The `.mat` file generated from the SEM analysis can be loaded in MATLAB. Modify the MATLAB script (`visualize_in_matlab.m`) to suit your visualization requirements.
