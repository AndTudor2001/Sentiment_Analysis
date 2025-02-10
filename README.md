Sentiment Analysis Project
This repository contains a comprehensive sentiment analysis project that utilizes various machine learning models and natural language processing (NLP) techniques to classify text data based on sentiment.

Project Overview
The project is divided into two main approaches:

TF-IDF with Classical Machine Learning Models: This approach involves text preprocessing, feature extraction using TF-IDF, and classification using models such as Logistic Regression, Random Forest, and Naive Bayes.

BERT Embeddings with Deep Learning Models: This approach leverages pre-trained BERT embeddings to capture contextual information from text, followed by classification using models like Logistic Regression and Random Forest.

Additionally, the project includes a Streamlit-based web application that allows users to input text and receive sentiment predictions in real-time.

Repository Structure
Project_TF-IDF.py: Contains the implementation of the TF-IDF approach with classical machine learning models.
Project_BERT.py: Contains the implementation of the BERT embeddings approach with deep learning models.
text.csv: The dataset used for training and evaluation.
README.md: This file, providing an overview of the project.
Dataset
The dataset (text.csv) consists of text entries labeled with corresponding sentiment categories. It is essential to ensure that the dataset is preprocessed appropriately before training the models.

Requirements
To run the code in this repository, you'll need the following Python packages:

pandas
numpy
scikit-learn
nltk
scipy
streamlit
matplotlib
spacy
transformers
torch
textblob
