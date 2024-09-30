# Consumer Survey Data Analysis using Structural Equation Modeling (SEM)

## Overview

This repository demonstrates how to analyze consumer survey data using **Structural Equation Modeling (SEM)** and visualize the results using **MATLAB**. The provided scripts are adaptable and can be customized to fit various survey data and research objectives.

## Project Overview

This repository outlines a comprehensive workflow for conducting SEM analysis on consumer survey data, including:

1. **Data Preprocessing**: Handling both categorical and numerical data, applying necessary mappings, and performing normalization.
2. **SEM Analysis**: Setting up and executing SEM analysis using the `semopy` package in Python.
3. **MATLAB Visualization**: Visualizing SEM results, such as path diagrams and other relevant visualizations, using MATLAB.

The repository is organized into the following key sections:

- **Data Preprocessing and SEM Analysis**
- **Visualizing SEM Results in MATLAB**
- **Customization Guide** for modifying the scripts to suit different datasets.

## Getting Started

### Prerequisites

To run the scripts effectively, ensure you have the following:

1. **Python Libraries**
   - `pandas`
   - `numpy`
   - `semopy`
   - `matplotlib`
   - `seaborn`
   - `scipy`
   - `scikit-learn`

2. **MATLAB** (for visualizing SEM results)
   - Basic knowledge of handling `.mat` files in MATLAB.
   - MATLAB installed with any necessary visualization toolboxes.

### Installation

1. Install the required Python libraries:

    ```bash
    pip install pandas numpy semopy matplotlib seaborn scipy scikit-learn
    ```

2. Ensure MATLAB is set up on your system for visualizing the `.mat` files generated during analysis.

### Data Preparation

1. Place your dataset in the designated data folder. The dataset should be formatted as a `.csv` file.
2. Ensure that the dataset is structured correctly with appropriate column headers and data types.

### How to Use

#### Step 1: Preprocess and Analyze Data using Python

- Open the analysis script and modify it to fit your survey dataset, including model specifications and dataset paths.
- Run the script to perform SEM analysis and save the results for MATLAB visualization:

- The script will generate SEM results in `.mat` format (MATLAB file) for further visualization.

#### Step 2: Visualize SEM Results using MATLAB

- Open MATLAB and navigate to the appropriate folder containing the generated scripts.
- Load and modify the MATLAB visualization script to meet your analysis needs.
- Run the MATLAB script to visualize SEM results, generating path diagrams and other visualizations as specified.

## Customization Guide

### Python Script Customization

1. **Preprocessing & Data Mapping**
   - Customize the mappings in the preprocessing script to align with the categories in your dataset.
   - Modify the list of numerical columns to include specific variables relevant to your analysis.

2. **SEM Analysis**
   - Update the SEM model specification according to your theoretical framework and variables of interest.
   - Adjust the column names and relationships as needed.

### MATLAB Script Customization

1. **Loading Data**
   - Ensure the `.mat` file generated from the SEM analysis is correctly referenced in the MATLAB script.

2. **Visualization Parameters**
   - Modify the visual elements and path diagrams based on your specific research and visualization requirements.

The resulting SEM analysis and MATLAB visualizations will provide valuable insights into relationships within your data, which can be utilized for academic research, industry reports, or strategic business decisions.

## FAQ

1. **How can I adjust the SEM model?**
   - You can modify the model specifications in the SEM analysis script to define your variables and relationships according to your research framework.

2. **Is this repository suitable for other survey data?**
   - Yes! Simply replace the dataset and adjust the mappings, SEM model, and visualizations to fit your specific needs.

3. **How do I visualize the results in MATLAB?**
   - Load the generated `.mat` file in MATLAB and adapt the MATLAB script to fit your visualization requirements. 
