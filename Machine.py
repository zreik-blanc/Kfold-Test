### Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracies = []
precisions = []
recalls = []
f1s = []

train = pd.read_csv('train.csv') # Load the Titanic training dataset
test = pd.read_csv("test.csv")
passenger_ids = test["PassengerId"] # Store Passenger IDs for submission later

### Data Preprocessing
def clean_data(dataset):
    """
    This function cleans and reorganizes the Titanic dataset for analysis.
    The steps include:
    1. Extracting titles from names and groups rare titles together into 'Rare'
    2. Calculating 'FamilySize' by summing 'SibSp' + 'Parch' + 1 (himself) and categorizes it into:
       - 'Single': 1 member
       - 'Couple': 2 members
       - 'Intermediate': 3-4 members
       - 'Large': 5+ members
    3. Filling missing 'Embarked' values with the most frequent port because its not that important in the prediction.
    4. Filling missing 'Age' values with the median age.
    5. Dropping unnecessary columns: 'PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'.
    This function returns:
        - A clean dataset ready for encoding and model training.
    """
    # Clean the unnecesarry titles and group them into 'Rare'
    dataset_title = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']] # Braund, Mr. Owen Harris -> Mr.
    dataset['Title'] = pd.Series(dataset_title) # Add the titles to the new column 'Title'
    dataset["Title"].value_counts() # Check the value counts of the titles
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')
    dataset["FamilySize"] = dataset['SibSp'] + dataset['Parch'] + 1

    def count_family(x):
        if x < 2:
            return 'Single'
        elif x == 2:
            return 'Couple'
        elif x <= 4:
            return 'Intermediate'
        else:
            return 'Large'
        
    dataset['FamilySize'] = dataset['FamilySize'].apply(count_family) # Adds the 'FamilySize' column based on the number of family members
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0]) # Fills missing 'Embarked' values with the most frequent port
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median()) # Fills missing 'Age' values with the median age
    dataset = dataset.drop(['PassengerId', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1) # Drops unnecessary columns because we turned them into features or they are not useful for the model
    return dataset


def encode_data(X):
    # This function encodes categorical variables in the dataset using one-hot encoding.
    # It converts categorical columns into binary columns, allowing the model to interpret them correctly.
    categorical_columns = ['Pclass','Sex', 'FamilySize', 'Embarked', 'Title'] # list of the categorical columns that we want to encode

    X_enc = pd.get_dummies(X, prefix=categorical_columns, columns=categorical_columns, drop_first=True) # one-hot encoding of the categorical columns
    return X_enc

train_clean = clean_data(train)

# Exclude the survived column to make the machine actually predict
X = train_clean.iloc[:, 1:]
y = train_clean.iloc[:, 0]

X_enc = encode_data(X).astype(float)

test_clean = clean_data(test)
X_test = encode_data(test_clean).astype(float)

class Model(nn.Module):
    # Constructs a neural network model with two fully connected layers.
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(14, 270) # Fully connected layer with 14 input features (X_enc's Column number) and 270 output features
        self.fc2 = nn.Linear(270, 2) # Fully connected layer with 270 input features and 2 output features (for binary classification on survival)

    # Forward pass through the network
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1) # Dropout %10 from all of the layers to prevent overfitting
        x = self.fc2(x)

        return x

kf = KFold(n_splits=5, shuffle=True, random_state=42) # Initialize KFold cross-validation

for fold, (train_index, val_index) in enumerate(kf.split(X_enc)):
    print(f'Fold {fold+1}')

    x_train, x_val = X_enc.iloc[train_index], X_enc.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize the model
    model = Model()

    ### Model parameters
    batch_size = 50
    num_epochs = 50
    learning_rate = 0.01
    batch_no = len(x_train) // batch_size

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The model is trained for a specified number of epochs, using the Adam optimizer and cross-entropy loss function.
    for epoch in range(num_epochs):
        x_train, y_train = shuffle(x_train, y_train)

        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size

            x_var = torch.tensor(x_train.values[start:end], dtype=torch.float32)
            y_var = torch.tensor(y_train.values[start:end], dtype=torch.long)


            optimizer.zero_grad() # Clear the gradients of all optimized tensors
            ypred_var = model(x_var) # Forward pass: compute predicted y by passing x to the model
            loss = criterion(ypred_var, y_var) # Calculate the loss using the predicted and actual values
            loss.backward() # Backpropagation: compute gradient of the loss with respect to model parameters
            optimizer.step() # Update the model parameters based on the gradients

    validation_data = torch.FloatTensor(x_val.values)
    # We do not need to calculate gradients during validation, so we use torch.no_grad()
    with torch.no_grad():
        result = model(validation_data)

    values, labels = torch.max(result, 1)
    labels_np = labels.numpy()

    accuracies.append(accuracy_score(y_val, labels_np)) # Calculate accuracy
    precisions.append(precision_score(y_val, labels_np))
    recalls.append(recall_score(y_val, labels_np)) # Calculate recall
    f1s.append(f1_score(y_val, labels_np)) # Calculate F1 score

print(f'Accuracy: {np.mean(accuracies):.2f}')
print(f'Precision: {np.mean(precisions):.2f}')
print(f'Recall: {np.mean(recalls):.2f}')
print(f'F1-score: {np.mean(f1s):.2f}')
print(f'Number of correct predictions: {np.sum(labels_np == y_val)}')
print(f'Number of wrong predictions: {len(y_val) - np.sum(labels_np == y_val)}')

model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test.values)
    result = model(X_test_tensor)
    _, labels_test = torch.max(result, 1)
    labels_test_np = labels_test.numpy()

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": labels_test_np
})

submission.to_csv("submission.csv", index=False)