# Real-or-Fake-NLP-with-X-Tweets

Project Name: NLP DISASTER TWEETS: EDA, NLP, TENSORFLOW, KERAS
Problem Description:
Sentiment Analysis of the dataset of twitter disaster tweets and predicting
  Actual Disaster
  Metaphorically Disaster

Table of Contents:
Introduction
Libraries
Loading Data
Exploratory Data Analysis
   Analyzing Labels
   Analyzing Features
      Sentence Length Analysis
Data Cleaning
    Remove URL
   Handle Tags
   Handle Emoji
   Remove HTML Tags
   Remove Stopwords and Stemming
   Remove Useless Characters
   WORLDCLOUD
Final Pre-Processing Data
Machine Learning
   Logistic Regression
   Navie Bayes
      Gaussian Naive Bayes
      Bernoulli Naive Bayes
      Complement Naive Bayes
      Multinomial Naive Bayes
   Support Vector Machine (SVM)
      RBF kernel SVM
      Linear Kernel SVM
   Random Forest
Deep Learning
   Single Layer Perceptron
   Multi Layer Perceptron
      Model 1 : SIGMOID + ADAM
      Model 2 : SIGMOID + SGD
      Model 3 : RELU + ADAM
      Model 4 : RELU + SGD
      Model 5 : SIGMOID + BATCH NORMALIZATION + ADAM
      Model 6 : SIGMOID + BATCH NORMALIZATION + SGD
      Model 7 : RELU + DROPOUT + ADAM
      Model 8 : RELU + DROPOUT + SGD

Pre-requisites and Installation:
This project requires Python and the following Python libraries installed:
   NumPy
   Pandas
   Matplotlib
   scikit-learn
   Tensorflow
   Keras

Requirements

Data Overview
Size of tweets.csv - 1.53MB
Number of rows in tweets.csv = 11369
Features:
     id - a unique identifier for each tweet
     text - the text of the tweet
     location - the location the tweet was sent from (may be blank)
     keyword - a particular keyword from the tweet (may be blank)
     target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

WordCloud
Word Clouds are a visual representation of the frequency of words within a given tweets. Word Cloud

Results
Key Performance Index:
Micro f1 score: Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.
Macro f1 score: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
Micro-Averaged F1-Score (Mean F Score): The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
F1 = 2 (precision recall) / (precision + recall)

All the models are compared on the basis of Accuracy, Precision, Recall, F1-Score, Time.

Results

Best Performing Models are: - Support Vector Machine, Deep Learning(Relu + Adam), Deep Learning(Relu + Adam + Dropouts)

Conclusion
Deep Learning Models are easy to overfit and underfit.
Do not underestimate the power of Machine Learning techniques.
Relu and Adam with Dropout proved to best as expected.
SVM is still the best as far as accuracy and training time is concerned.

References:
https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data
https://towardsdatascience.com/natural-language-processing-nlp-for-machine-learning-d44498845d5b
https://machinelearningmastery.com/
