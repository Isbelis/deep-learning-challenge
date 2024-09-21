# Deep Learning Challenge: : Alphabet Soup Nonprofit Foundation

This deep learning challenge involved analyzing a dataset from the nonprofit foundation, Alphabet Soup, to create a binary classifier that predicts which applicants for funding have the best chance of success in their ventures.
The report highlights the variables and details about the data that were crucial during the analysis.

## Key Steps to Completing the Challenge:

### Step 1: Preprocess the Data

1. Loaded the `charity_data.csv` into a Pandas DataFrame and identified:
   - The **target variable** for the model.
   - The **feature variable(s)** for the model.
     
2. Dropped the `EIN` and `NAME` columns.
3. Determined the number of unique values for each column.
4. For columns with more than 10 unique values, identified the number of data points for each unique value.
5. Based on the number of data points, combined "rare" categorical variables into a new category, "Other", and verified the replacement was successful.
6. Encoded categorical variables using `pd.get_dummies()`.
7. Split the preprocessed data into a feature array, `X`, and a target array, `y`. Then, used the `train_test_split` function to divide the data into training and testing sets.
8. Scaled the training and testing feature datasets by creating a `StandardScaler` instance, fitting it to the training data, and transforming the data.

### Step 2: Compile, Train, and Evaluate the Model
1. Continued using Google Colab to perform the preprocessing steps from Step 1.
2.Created a neural network model using **TensorFlow** and **Keras**, assigning the number of input features and nodes for each layer.
3. Designed **two hidden layers** and selected appropriate activation functions (Model 1).
4. Added an output layer with the appropriate activation function.
5. Verified the structure of the model.
6. Compiled and trained the model.
7. Created a callback to save the model's weights every five epochs.
8. Evaluated the model using the test data to determine loss and accuracy.
10. Saved and exported the results to an HDF5 file, named `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model
**Objective:** Optimize the model to achieve a target predictive accuracy of over 75%.

In this step, the following actions were taken:
1. Added the variable NAME, except in the Keras Tuner optimized model.
2. Reduced the number of **epochs** in one model.
3. Added or reduced the number of neurons in the hidden layers.
4. Increased the number of hidden layers.
5. Tested different **activation functions** for the hidden layers **(Tanh, ReLU, Sigmoid)**.

**Note:** Over **12 models** were tested, both with and without the `NAME` variable. The final set of four models, including the initial unoptimized one, displayed the highest accuracy.

## Alternative Machine Learning Models
While the challenge required a deep learning approach, I also explored alternative machine learning models to compare the results. Specifically, I used **Random Forest Classifier** and **XGBoost Classifier**, which provided valuable insights in terms of accuracy and model performance.

### 1. Random Forest Classifier:

- This ensemble learning method combines multiple decision trees to improve predictive accuracy. It was particularly useful for dealing with the imbalanced dataset and helped reduce overfitting by averaging the predictions from different trees.
  
- The model demonstrated robust performance, making it a good alternative for binary classification tasks.

### 2. XGBoost Classifier:

- XGBoost is a powerful implementation of gradient boosting that builds models sequentially by optimizing the residuals from previous models. It showed strong performance on the tabular data used in this challenge.

- The model was effective in handling the complexity of the dataset and achieved competitive results compared to the neural network models.

These two models provided interpretable and fast-to-train alternatives to deep learning. They were useful for validating results and optimizing overall model performance. By comparing the results of these models with the deep learning approach, I was able to assess the strengths of both techniques in terms of accuracy, training time, and computational efficiency.

## Resources and Tools:
- *Pandas*
- *scikit-learn*
- *Google Colab*
- *NumPy*
- *TensorFlow*
- *Keras*
  
## Folder and Files:

**- Model_h5:** Contains all models in HDF5 format.

**- Model_notebook:** Contains three Jupyter notebooks:

  *1. AlphabetSoupCharity.ipynb:* Steps 1 and 2.
  
  *2. AlphabetSoupCharity_Optimization_kerasturner.ipynb:* Steps 1, 2, and 3.
  
  *3. AlphabetSoupCharity_Optimization.ipynb:* Adjusted Step 1, and Steps 2 and 3.
  
**- Output:** Includes all graphs related to accuracy and loss.


### File:

**AlphabetSoupCharity_report.md:** Contains the report and analysis details.

**Note:** All folders and files are exported from Google Colab, while the report file is independent of this origin.

## Reference:

  - [Bootcamp Spot Assignment] (https://bootcampspot.instructure.com/courses/6446/assignments/78963?module_item_id=1249103 1/6)
  - IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/ (https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)

