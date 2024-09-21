# Alphabet Soup Charity Report

## Overview of the Analysis
The purpose of this analysis is to develop machine learning and neural networks that can predict whether applicants will be successful if funded by Alphabet Soup. In other words, the challenge is to provide a tool that can help Alphabet Soup select the applicants with the best chance of success in their ventures.

### Data Overview
The dataset contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. The metadata for each organization has been captured with the following columns:

**EIN and NAME:** Identification columns

**APPLICATION_TYPE:** Alphabet Soup application type

**AFFILIATION:** Affiliated sector of industry

**CLASSIFICATION:** Government organization classification

**USE_CASE:** Use case for funding

**ORGANIZATION:** Organization type

**STATUS:** Active status

**INCOME_AMT:** Income classification

**SPECIAL_CONSIDERATIONS:** Special considerations for application

**ASK_AMT:** Funding amount requested

**IS_SUCCESSFUL:** Whether the money was used effectively

## Machine Learning Process
The process for building and evaluating binary classifier models included the following stages:

### 1. Data Preprocessing:
#### 1. Target Variable:
The target variable for this model is IS_SUCCESSFUL.

   *- 0:* Applicants' funding not successful (16,038 instances)
   
   *- 1:* Applicants' funding successful (18,261 instances)
   
#### 2. Feature Variables:

The feature variables vary depending on the step:

   **- Step 2: Compile, Train, and Evaluate the Model:**  The features are `APPLICATION_TYPE`, `AFFILIATION`,`CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
   
   **- Step 3: Optimize the Model:** The features are the same as Step 2, with the addition of `NAME`. In the Keras Turner Optimization Model, the `NAME` variable was excluded.
   
#### 3. Dropped Columns:

The columns **`EIN`** and **`NAME`** were dropped to preprocess the data, specifically in Step 2 and the Keras Turner Optimization Model.


### 2. Compiling, Training, and Evaluating the Model:

In this phase, neural network models were created using early stopping to prevent overfitting.

#### Model 1:

   **- First Hidden Layer:** 100 neurons with ReLU activation
   
   **- Second Hidden Layer:** 50 neurons with ReLU activation
   
   **- Output Layer:** 1 neuron with Sigmoid activation

   **- Epochs:** 200
   
   **- Accuracy:** 0.7262
   
   **- Loss:** 0.5550
   
   **- Model Saved as:** 'AlphabetSoupCharity.h5'
   
**- Justification:** The ReLU activation function in the hidden layers helps the model learn complex relationships, while the Sigmoid function in the output layer is suitable for binary classification tasks.


#### Model 2 (Keras Turner):

*Hyperparameters:*
- First Units: 9
  
- Number of Layers: 3
- Units: 9, 3, 3
- Activation: Sigmoid
- Epochs: 20
  
**- Accuracy:** 0.7289

**- Loss:** 0.5634

![Model_Evaluation_Keras](https://github.com/user-attachments/assets/e23f6785-f40b-4946-927a-eda0a73756a8)

![Training_Validation_AccuracyKeras](https://github.com/user-attachments/assets/2707ecf6-43f1-4ca1-b044-f9dade975c3e)

![Training_Validation_LossKeras](https://github.com/user-attachments/assets/d617e2e2-c956-4621-987e-e76d6a44c1fd)

**- Model Saved as:** 'AlphabetSoupCharity_Optimization_2_KERAS.h5'
      
**- Justification:** This model used Keras Turner to optimize hyperparameters. The layers and activations were adjusted to improve the performance and structure of the model.


#### Model 3:

   **- First Hidden Layer:** 100 neurons with ReLU activation
   
   **- Second Hidden Layer:** 100 neurons with ReLU activation
   
   **- Third Hidden Layer:** 50 neurons with ReLU activation
   
   **- Fourth Hidden Layer:** 20 neurons with ReLU activation
   
   **- Output Layer:** 1 neuron with Sigmoid activation
   
   **- Epochs:** 100
   
   **- Accuracy:** 0.7562
   
   **- Loss:** 0.4906

   ![Model_Accuracy_3](https://github.com/user-attachments/assets/f8919266-73d3-46f8-a004-c2cb2a7911ce)

  ![Model_loss_3](https://github.com/user-attachments/assets/1882bc83-0175-46e5-bf25-dac53c2098b3)

   **- Model Saved as:** 'AlphabetSoupCharity_Optimization_3.h5'
   
**Justification:** This model focused on optimizing the number of hidden layers and neurons while adding the `NAME` column. This configuration helped improve accuracy.


#### Model 4:
  **- First Hidden Layer:** 20 neurons with ReLU activation
  
  **- Second Hidden Layer:** 10 neurons with Tanh activation
  
  **- Third Hidden Layer:** 5 neurons with Tanh activation
  
  **- Output Layer:** 1 neuron with Sigmoid activation
  
  **- Epochs:** 50
  
  **- Accuracy:** 0.7541
  
  **- Loss:** 0.4908
  
  ![Model_Accuracy_4](https://github.com/user-attachments/assets/52d8d4b5-3ed2-4d97-b334-7e69024d5cde)

  ![Model_loss_4](https://github.com/user-attachments/assets/ca700076-2f47-4ad9-992e-f35acfaf660c)


  **- Model Saved as:** 'AlphabetSoupCharity_Optimization_4.h5'

**- Justification:** This model reduced the number of hidden layers and neurons. The choice of activation functions **(ReLU and Tanh)** was based on the outputs from Keras Turner. Adding the `NAME` column contributed to achieving higher accuracy.


### Overall Interpretation and Summary of Results:

This analysis aimed to predict the success of applicants funded by Alphabet Soup using neural network models. The dataset comprised over 34,000 organizations, with various metadata features. The target variable was binary (whether the funding was successful or not), and several machine learning models were trained to create a classifier capable of predicting success.

#### Model Results:
1. **Model 1** served as the baseline, achieving an accuracy of **72.62%**. While it performed moderately well, the high number of neurons in the hidden layers led to concerns about overfitting, without delivering significant gains in accuracy.
   
2. **Model 2 (Keras Turner)** improved the model slightly by optimizing hyperparameters, achieving an accuracy of **72.89%**. This model used fewer neurons but did not achieve notable improvements in performance, indicating that more sophisticated tuning might be necessary to enhance accuracy further.

3. **Model 3** introduced a deeper architecture with four hidden layers and achieved an accuracy of **75.62%**. This model's complex structure allowed it to capture more detailed patterns in the data, making it the best-performing model so far.

4. **Model 4**reduced the number of layers and neurons while using both **ReLU** and **Tanh** activations. It achieved a comparable accuracy of **75.41%**, demonstrating that a simpler model can perform just as well as the more complex Model 3, making it more computationally efficient.
   
#### Overall Interpretation:

Both **Model 3** and **Model 4** delivered the best results, with accuracies close to **76%**. Model 3’s deeper structure captured more complexity, while Model 4’s simplicity and fewer layers made it an efficient alternative without sacrificing performance. Despite these gains, the accuracy plateaued around 76%, indicating the model’s limit using the current dataset and architecture.

Further improvements could be explored using alternative optimization methods, additional feature engineering, or different machine learning models to break past the 75% accuracy ceiling.

### Response to Further Analysis:
Given the moderate performance of neural networks, other models such as **Random Forest Classifier** and **XGBoost Classifier** could provide better results, particularly with large datasets like the one used here:

- **Random Forest Classifier** yielded an accuracy of **74%** with good recall and precision values, making it a strong candidate for predicting success.

    - **Accuracy:** 74%
    - **Precision (Class 1):** 0.73
    - **Recall (Class 1):** 0.81
    - **AUC:** 0.7993
    - **f1-score (Class 1):** 0.77
      
- **XGBoost Classifier**, often known for handling large datasets efficiently, produced an accuracy of **75%**, similar to the neural networks but with a higher recall.

    - **Accuracy:** 75%
    - **Precision (Class 1):** 0.73
    - **Recall (Class 1):** 0.85
    - **AUC:** 0.8178
    - **f1-score (Class 1):** 0.79
 
## Conclusion:
Both **neural networks** and **alternative machine learning models** like Random Forest and XGBoost performed similarly, with XGBoost demonstrating slightly better recall and AUC values. As a recommendation, **Models 3** and **4** can be selected for predicting applicant success, but incorporating Random Forest or XGBoost may further improve model performance by handling complex features more efficiently.
