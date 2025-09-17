# Titanic Survival Prediction with PyTorch Neural Network

A machine learning project that predicts passenger survival on the Titanic using a PyTorch neural network with k-fold cross-validation for robust model evaluation.

## Overview

This project implements a deep learning solution for the classic Titanic dataset, featuring comprehensive data preprocessing, feature engineering, and a neural network trained with cross-validation to predict passenger survival rates.

## Features

- **Advanced Data Preprocessing**: Title extraction, family size categorization, and missing value imputation
- **Feature Engineering**: Creates meaningful features like family size groups and title categories
- **Neural Network Architecture**: Two-layer fully connected network with dropout regularization
- **K-Fold Cross-Validation**: 5-fold validation for robust performance evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score reporting

## Dataset

The project uses the famous Titanic dataset containing passenger information including:
- Demographics (age, sex, class)
- Family relationships (siblings, spouses, parents, children)
- Ticket and cabin information
- Port of embarkation

## Model Architecture

```
Input Layer (14 features) → Hidden Layer (270 neurons) → Output Layer (2 classes)
                          ReLU Activation              Binary Classification
                          Dropout (0.1)
```

## Data Preprocessing Pipeline

### 1. Title Extraction
- Extracts titles from passenger names (Mr., Mrs., Miss., etc.)
- Groups rare titles into a single 'Rare' category

### 2. Family Size Engineering
- Calculates total family size: `SibSp + Parch + 1`
- Categorizes into groups:
  - **Single**: 1 member
  - **Couple**: 2 members  
  - **Intermediate**: 3-4 members
  - **Large**: 5+ members

### 3. Missing Value Treatment
- **Age**: Filled with median age
- **Embarked**: Filled with most frequent port

### 4. Feature Encoding
- One-hot encoding for categorical variables
- Drops unnecessary columns (PassengerId, Cabin, Name, SibSp, Parch, Ticket)

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
scikit-learn>=1.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-pytorch-prediction.git
cd titanic-pytorch-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the Titanic dataset files:
   - `train.csv`
   - `test.csv`

## Usage

Run the main script to train the model and generate predictions:

```bash
python Machine.py
```

The script will:
1. Load and preprocess the training data
2. Perform 5-fold cross-validation
3. Display comprehensive performance metrics
4. Generate predictions for the test set
5. Create a `submission.csv` file for Kaggle submission

## Model Performance

The model is evaluated using k-fold cross-validation with the following metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision and recall

## Hyperparameters

- **Batch Size**: 50
- **Epochs**: 50
- **Learning Rate**: 0.01
- **Hidden Layer Size**: 270 neurons
- **Dropout Rate**: 0.1
- **Cross-Validation Folds**: 5

## File Structure

```
├── Machine.py    # Main script
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── submission.csv          # Generated predictions
└── README.md              # Project documentation
```

## Key Functions

- `clean_data(dataset)`: Comprehensive data cleaning and feature engineering
- `encode_data(X)`: One-hot encoding for categorical variables
- `Model`: PyTorch neural network class with forward pass implementation

## Results

The model outputs:
- Cross-validation performance metrics
- Detailed prediction counts (correct vs incorrect)
- Kaggle-ready submission file


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the Titanic dataset
- PyTorch team for the deep learning framework
- Scikit-learn for preprocessing and evaluation utilities

## Future Improvements

- [ ] Hyperparameter tuning with grid search
- [ ] Ensemble methods integration
- [ ] Advanced feature engineering
- [ ] Model interpretability analysis
- [ ] Automated model selection
