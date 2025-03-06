# AgingDrugOmics

**AgingDrugOmics** uses both metabolomic and proteomic plasma data to predict whether a 12-month-old UM-HET3 male mouse has been given an anti-aging intervention. The machine-learning model employed is **XGBoost**, and the analysis requires both **R** and **Python**.

> **Note:** Access to the raw data (Step 1) will be provided upon manuscript acceptance.

---

## Overview

- **Goal**: Discriminate between control and slow-aging (anti-aging treated) mice using metabolomic and proteomic datasets.  
- **Highlights**:  
  - Normalizes and combines multiple omics datasets.  
  - Performs 10k cross-fold validation.  
  - Tests on novel interventions excluded from training data.  
  - Generates SHAP values and visualizations for model interpretability.  

This repository accompanies the manuscript titled:  
**“[Discrimination of Normal from Slow-Aging Mice by Plasma Metabolomic and
Proteomic Features – Link will be added upon publication]”**  

---

## System Requirements

- **Operating System**:  
  - Tested on Windows 11 (version 24H2)  

- **Software**:  
  - **R** (Recommended version: 4.0+)  
  - **Python** (Recommended version: 3.8+)  
  - **RStudio** and/or **Anaconda (Jupyter Lab)** for notebook execution  

---

## Dependencies

A detailed list of required packages for each step of the pipeline is available in the [Package_Info folder](https://github.com/BrettonB/AgingDrugOmics/tree/main/Paper_Figures/Package_Info).

---

## How to Run the Code

1. **Download the Repository**  
   - Click the green “Code” button and select “Download ZIP” or clone via Git:
     ```bash
     git clone https://github.com/BrettonB/AgingDrugOmics.git
     ```
   - Navigate into the directory:
     ```bash
     cd AgingDrugOmics
     ```

2. **Prepare Your Environment**  
   - **Python**: install anaconda (https://www.anaconda.com/products/navigator) and start Jupyter Lab and install the dependencies in [Package_Info folder](https://github.com/BrettonB/AgingDrugOmics/tree/main/Paper_Figures/Package_Info).
   - **R**: Install R studio (https://posit.co/download/rstudio-desktop/) and install the dependencies in [Package_Info folder](https://github.com/BrettonB/AgingDrugOmics/tree/main/Paper_Figures/Package_Info).

3. **Follow the Steps Sequentially**  
   Each step in the analysis pipeline is labeled **Step#...** (from Step 1 through Step 13).  
   - Start with **Step 1** and proceed in numerical order.  
   - Within each step’s folder, there is either an **.Rmd** file (R Markdown) or an **.ipynb** file (Jupyter Notebook).  
   - **Open and run** the corresponding `.Rmd` or `.ipynb` in **RStudio** or **Jupyter Lab**. These files will orchestrate the associated script files.

4. **Check the Output**  
   - All relevant outputs (figures, intermediate data, final results) will be saved in the `Paper_Figures` folder (and subfolders within it).  
   - Log or console messages will guide you on any required input data formats or additional steps.

---

## High-Level Code Functionality

1. **Input Data**  
   - **Metabolomic data** for control vs. anti-aging–treated mice (named metabolites, not raw peaks).  
   - **Proteomic data** for control vs. anti-aging–treated mice (named proteins, not raw peaks).  
   - The abovde datasets include metadata describing intervention groups, sex, sample IDs, and the metabolites.

2. **Data Processing & Modeling**  
   - **Normalization and Integration**: Merges metabolomic and proteomic data into a unified dataset.  
   - **Cross-Fold Validation**: Performs 10K cross-fold splits for robust performance estimation.  
   - **Novel Intervention Testing**: Excludes one intervention group from training to assess predictive performance on truly novel data.  
   - **Feature Importance (SHAP)**: Calculates SHAP values for model interpretability.

3. **Output**  
   - **Normalized Datasets**: Cleaned and combined metabolomic/proteomic matrices.  
   - **Prediction Results**: statistics for the cross-fold validation and novel-intervention tests.  
   - **SHAP Interpretations**: Feature importance rankings and visual explanations for model decisions.  
   - **Visualizations**: Plots and figures demonstrating performance, data distributions, and SHAP results.

---

## Future Updates

- **Raw Data**: Will be uploaded following the acceptance of the associated manuscript.  
- **Paper Link**: The README will be updated once the paper is published and publicly accessible.  

---

## Contact

For questions or collaboration inquiries, please reach out via the [GitHub Issues](https://github.com/BrettonB/AgingDrugOmics/issues) section or email **[brettonb@umich.edu]**.

---
