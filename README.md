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


![image](https://github.com/user-attachments/assets/b15bbb6b-5999-4658-991d-8082b61be88c) 



    # Determine the number of unique values in each column.
application_df.nunique()


![image](https://github.com/user-attachments/assets/5d9c87ac-1730-497a-b8fe-c5c34808b977)


    # Look at APPLICATION_TYPE value counts to identify and replace with "Other"
application_df['APPLICATION_TYPE'].value_counts()



![image](https://github.com/user-attachments/assets/821e0762-92ac-4d20-8b3e-b0f2b660c08a)  



    

    Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

    # Choose a cutoff value and create a list of classifications to be replaced
    # use the variable name `classifications_to_replace`
    
   classifications_to_replace = application_df['CLASSIFICATION'].value_counts().loc[lambda x : x<1883].index.tolist() 
   
# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")

# Check to make sure replacement was successful
application_df['CLASSIFICATION'].value_counts()



![image](https://github.com/user-attachments/assets/4229bf94-e0a6-47b6-92de-12853d22bed7)


    

    Use pd.get_dummies() to encode categorical variables.

# Convert categorical data to numeric with `pd.get_dummies`
application_df = pd.get_dummies(application_df,dtype=float)
application_df.head() 


    Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

    # Split our preprocessed data into our features and target arrays
y=application_df['IS_SUCCESSFUL'].values
X=application_df.drop('IS_SUCCESSFUL', axis=1).values

    # Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42) 


    
    # Split our preprocessed data into our features and target arrays
y=application_df['IS_SUCCESSFUL'].values
X=application_df.drop('IS_SUCCESSFUL', axis=1).values

    # Split the preprocessed data into a training and testing dataset
# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42) 

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

    # Create a StandardScaler instances
scaler = StandardScaler()

    # Fit the StandardScaler
X_scaler = scaler.fit(X_train)

    # Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)  

    # Display the data

print(X_train_scaled)

![image](https://github.com/user-attachments/assets/a27ccdbf-3cb3-4f4d-885b-e07d09ca9294)



Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

    Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
    
    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
      
      Create a callback that saves the model's weights every five epochs.

      Evaluate the model using the test data to determine the loss and accuracy.

      Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.


input_features = len(X_train_scaled[0])

model = tf.keras.models.Sequential()

    # First hidden layer
model.add(tf.keras.layers.Dense(units=80, activation="relu", input_dim=input_features))

    # Second hidden layer
model.add(tf.keras.layers.Dense(units=30, activation="relu"))

    # Output layer
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Check the structure of the model
model.summary()  

![image](https://github.com/user-attachments/assets/7e7a2680-352c-4993-b168-32fb3388a2ec)

    
    
    

    # Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  

    # Train the model
model.fit(X_train_scaled, y_train, epochs=100)  

    # Evaluate the model using the test data
model_loss, model_accuracy = model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


![image](https://github.com/user-attachments/assets/660b5f71-083e-4181-8826-979bd932b9d6)

    # Export our model to HDF5 file
model.save('AlphabetSoupCharity.h5') 


    # first optimization attempt design the neural network 
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    
    # Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with more epochs
fit_model = model.fit(X_train_scaled, y_train, epochs=150, validation_data=(X_test_scaled, y_test))

    # Evaluate the model
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Loss: {model_loss}, Model Accuracy: {model_accuracy}")

    # Second Optimization attempt
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='tanh', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    # Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with more epochs
fit_model = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))

    # Evaluate the model
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Loss: {model_loss}, Model Accuracy: {model_accuracy}")
    

    # Third optimization attempt
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='tanh', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

    # Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with more epochs
fit_model = model.fit(X_train_scaled, y_train, epochs=175, validation_data=(X_test_scaled, y_test))

    # Evaluate the model
model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Loss: {model_loss}, Model Accuracy: {model_accuracy}")


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


    # Export our model to HDF5 file
model.save('AlphabetSoupCharity_Optimization.h5')












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


