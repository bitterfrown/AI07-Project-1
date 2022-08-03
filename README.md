# Project 1: Predicting Heart Disease Using Feed Forward Neural Network

## 1.0 Summary 
This project has been carried out to study the possibility of a patient to have heart disease using the highly accurate deep learning model. 😉

## 2.0 Framework
Spyder has been used in the process of loading the data, building and training the model. The libraries used in this project are Tensorflow Keras, Pandas, Numpy, Matplotlib, and Scikit Learn.

## 3.0 Methodology
### 3.1 Data Loading and Preparation
The data is loaded and preprocessed and checked for missing values. Then, the data is separated into train-validation-test sets, with a ratio of 80:10:10. Then, data normalization is executed by applying standard scaler method.  

### 3.2 Building the Model
Since this study leans more towards clasification problem, the neural network is designed to detect the presence of heart disease in each patient. 

### 3.3 Model Training
The model is trained with a batch size of 64 for epoch size of 100.

## 4.0 Result
After multiple attempts at increasing both the training accuracy and validation accuracy to reach 90%, the highest this model's accuracy could reach in order to obtain a decent looking graph is 86% and 85% for validation and training accuracy respectively. The accuracies can reach 90% but at the risk of producing a jagged looking graph. 


### 4.1 Training Accuracy vs Validation Accuracy
![Figure 2022-08-03 132109](https://user-images.githubusercontent.com/108327348/182531176-481955a9-dcae-437a-a218-3aed124dcf16.png)

### 4.2 Training Loss vs Validation Loss
![Figure 2022-08-03 132031](https://user-images.githubusercontent.com/108327348/182531290-ab9cd7f8-2898-453c-b704-2746f776361f.png)

### 4.3 Evaluation with Test Data

![Screenshot 2022-08-03 132151](https://user-images.githubusercontent.com/108327348/182531405-19504ece-3343-4613-ab85-f49579cc4a5b.jpg)
