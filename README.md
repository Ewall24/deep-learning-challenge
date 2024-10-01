# deep-learning-challenge 


# Neural Network Model for Alphabet Soup

Instructions
Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

    Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
        What variable(s) are the target(s) for your model?
        What variable(s) are the feature(s) for your model?  

 # Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf 
import warnings
warnings.filterwarnings('ignore')

#  Import and read the charity_data.csv.
import pandas as pd
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()       


![image](https://github.com/user-attachments/assets/c3def7df-e753-4c19-a876-de3dd90976fa)    



    Drop the EIN and NAME columns.

    # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df = application_df.drop(columns = ['EIN', 'NAME']) 
application_df.head() 




    Determine the number of unique values for each column.

    For columns that have more than 10 unique values, determine the number of data points for each unique value.

    Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

    Use pd.get_dummies() to encode categorical variables.

    Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

    Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.


Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

    Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

    Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

    Create the first hidden layer and choose an appropriate activation function.

    If necessary, add a second hidden layer with an appropriate activation function.

    Create an output layer with an appropriate activation function.

    Check the structure of the model.

    Compile and train the model.

    Create a callback that saves the model's weights every five epochs.

    Evaluate the model using the test data to determine the loss and accuracy.

    Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.


Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

    Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
        Dropping more or fewer columns.
        Creating more bins for rare occurrences in columns.
        Increasing or decreasing the number of values for each bin.
        Add more neurons to a hidden layer.
        Add more hidden layers.
        Use different activation functions for the hidden layers.
        Add or reduce the number of epochs to the training regimen.

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

    Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

    Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

    Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

    Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

    Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.















## Overview of the Analysis

The purpose of this analysis is to build and optimize a deep learning model that predicts whether organizations funded by Alphabet Soup will be successful. Using the charity dataset, we preprocess the data, train a binary classification model, and then optimize it to achieve higher than 75% accuracy. This model will help Alphabet Soup make data-driven decisions on which organizations to fund based on their likelihood of success.

---

## Results

### Data Preprocessing

- **Target Variable**:
  - The target variable is the `IS_SUCCESSFUL` column, which indicates if an organization was successful (1) or not (0).

- **Feature Variables**:
  - The feature variables are all columns that could affect the success of an organization, excluding irrelevant ones:
    - Application Type
    - Affiliation
    - Income Classification
    - Use Case
    - Funding Amount Requested
    - Special Conditions (e.g., Status, Organization Type)

- **Removed Variables**:
  - The `EIN` and `NAME` columns were removed, as they are unique identifiers with no predictive value.

- **Categorical Variables**:
  - For categorical columns with many unique values, rare categories were combined into an `"Other"` group to improve generalization.

- **Encoding and Scaling**:
  - Categorical variables were encoded using `pd.get_dummies()`.
  - The features were scaled using `StandardScaler()` to normalize the data.

### Compiling, Training, and Evaluating the Model

- **Model Architecture**:
  - **Neurons**: The model started with **80 neurons** in the first hidden layer and **30 neurons** in the second hidden layer.
  - **Layers**: Two hidden layers were used.
  - **Activation Functions**:
    - The **ReLU (Rectified Linear Unit)** activation function was used in the hidden layers.
    - The **Sigmoid** activation function was used in the output layer for binary classification.

- **Model Performance**:
  - **Accuracy**: The model achieved approximately **72.83% accuracy**.
  - **Loss**: The final loss was **0.5911**.
  
  While this accuracy is a decent start, it falls short of the desired 75% target.

- **Steps Taken to Improve Model Performance**:
  - **More Neurons**: Increased the number of neurons in the hidden layers to better capture complex patterns.
  - **Additional Layers**: Experimented with adding a third hidden layer.
  - **Epoch and Batch Size Tuning**: Adjusted the number of epochs and batch size to allow better learning.
  - **Early Stopping & Dropout**: Used dropout and early stopping techniques to avoid overfitting.

---

## Summary

### Overall Results

The model achieved **72.83% accuracy**, a reasonable outcome but below the target of 75%. The neural network architecture with two hidden layers and ReLU activation performed fairly well. However, additional optimization is required to achieve higher accuracy and model performance.

### Recommendations

To improve performance, the following steps are recommended:

1. **Hyperparameter Tuning**:
   - Continue fine-tuning hyperparameters such as the learning rate, batch size, and number of epochs to improve performance.
   
2. **Ensemble Methods**:
   - Explore alternative models like **Random Forest** or **Gradient Boosting**. These methods tend to perform better on tabular data and may yield higher accuracy.

3. **Feature Engineering**:
   - Consider advanced feature engineering techniques such as interaction terms or polynomial features to enrich the input data.

4. **Alternative Models**:
   - A simpler model like **Logistic Regression with Regularization** might perform better for this binary classification problem and offer easier interpretability.

---


