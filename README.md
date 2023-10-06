# LCK_QSAR
Quantitative Structure-Activity Relationship (QSAR) Modeling for Predicting LCK Inhibitory Activity

## Overview
LCK_QSAR is a machine learning project that aims to predict the inhibitory activity of compounds against Lymphocyte-specific protein tyrosine kinase (LCK). Utilizing a variety of regression algorithms and a stacking ensemble method, the project seeks to generate accurate and reliable predictive models based on molecular descriptors derived from compound structures (SMILES notation).

## Getting Started
Prerequisites
Ensure the following Python libraries are installed for running the project:

Pandas
Numpy
RDKit
Scikit-learn
XGBoost
LightGBM
You may install them using pip:


pip install pandas numpy rdkit scikit-learn xgboost lightgbm

The dataset should contain SMILES notation of compounds and their corresponding LCK inhibitory activity (pIC50 values).

### Usage
Data Preprocessing: Convert SMILES notation to numerical fingerprints and split data into training and test sets.
Model Training: Train individual regression models and an ensemble stacking model using the training data.
Hyperparameter Optimization: Employ grid search to fine-tune model parameters for improved predictive performance.
Model Evaluation: Validate the models using test data and evaluate their performance via metrics like R^2 score and RMSE.
### Code Structure
Data Preprocessing and Feature Generation: Convert molecular structures into hashed atom pair fingerprints.
Model Training and Validation: Implement and validate various regression models including Random Forest, XGBoost, K-Nearest Neighbors, LightGBM, and a stacking ensemble model.
Hyperparameter Optimization: Use grid search for finding optimal hyperparameters of the base models.
### Example Workflow
Load and preprocess data
Generate molecular fingerprints from SMILES notation
Train base regression models and a stacking model
Perform hyperparameter optimization
Evaluate model performance
(Optional) Deploy the model for predictions on new data

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgments
None
