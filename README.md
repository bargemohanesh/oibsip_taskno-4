# oibsip_taskno-4
Spam Classification Project

This project focuses on building a machine learning-based spam classifier using various algorithms, evaluating their performance, and identifying the best-performing model for predicting whether a message is spam or ham.

Table of Contents

Dataset

Dependencies

Preprocessing

Model Training

Model Evaluation

Best Model Selection

How to Use

Results

Dataset

File Path: spam.csv

Source: Kaggle spam dataset

Columns Used:

v1: Label (ham/spam)

v2: Message

Label Encoding:

ham = 0

spam = 1

Dependencies

The following libraries are required to run this project:

pandas

numpy

scikit-learn

nltk

imbalanced-learn

Use the following commands to install any missing libraries:

pip install pandas numpy scikit-learn nltk imbalanced-learn

Preprocessing

Steps:

Cleaning Text: Remove non-word characters, convert text to lowercase.

Tokenization: Split text into tokens.

Lemmatization: Reduce words to their base form using WordNetLemmatizer.

Stopword Removal: Exclude common English stopwords.

Vectorization: Convert text to numerical data using CountVectorizer.

Handling Class Imbalance: Use SMOTE (Synthetic Minority Oversampling Technique).

Model Training

The following algorithms were used:

Multinomial Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

Decision Tree Classifier

Each model was trained on the preprocessed dataset with class imbalance addressed using SMOTE.

Model Evaluation

Metrics:

Accuracy

Precision (Spam)

Recall (Spam)

F1-Score (Spam)

The performance of each model was evaluated using a separate test dataset. Metrics were calculated using classification_report and confusion_matrix.

Results:

Model

Accuracy

Precision (Spam)

Recall (Spam)

F1-Score (Spam)

Multinomial Naive Bayes

0.9731

0.87

0.93

0.90

Logistic Regression

0.9148

0.62

0.91

0.74

Support Vector Machine

0.8753

0.53

0.70

0.60

Random Forest Classifier

0.8897

0.55

0.93

0.69

Decision Tree Classifier

0.8529

0.47

0.91

0.62

Best Model Selection

A custom scoring function prioritized the following metrics (in order):

Recall (Spam)

Precision (Spam)

F1-Score (Spam)

Accuracy

Selected Model: Multinomial Naive Bayes

Metrics for the Best Model:

Accuracy: 0.9731

Precision (Spam): 0.87

Recall (Spam): 0.93

F1-Score (Spam): 0.90

How to Use

Predicting Messages

The predict_message function can classify new messages as spam or ham.

Example:

message1 = "Congratulations! You have won a $1000 gift card."
print(predict_message(message1))  # Output: Spam

message2 = "Let's discuss the project on Teams."
print(predict_message(message2))  # Output: Ham

Results

Best Model: Multinomial Naive Bayes

Example Predictions:

"Congratulations! You have won a $1000 gift card." -> Spam

"Let's discuss the project on Teams." -> Ham

For questions or feedback, please feel free to contact me.

