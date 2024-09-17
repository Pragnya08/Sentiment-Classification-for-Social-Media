# Sentiment-Classification-for-Social-Media

Project Overview:
This repository contains the implementation of multiple machine learning models to classify the sentiment of tweets as positive, negative, or neutral. The models are developed and tested using data from the SemEval 2017 Task 4 dataset, focusing specifically on Subtask A. The objective is to accurately classify the sentiment of each tweet using various traditional and deep learning approaches.

Data:
The dataset consists of tweets provided in a TSV (tab-separated values) format, including a tweet ID, sentiment label, and the tweet text. The data is divided into training, development, and multiple test sets, which are used to evaluate the models' performance across different scenarios.

Models Implemented:
Traditional Machine Learning Classifiers:

- Support Vector Machines (SVM): A linear SVM classifier trained on custom features extracted from the tweet text.
- Naive Bayes: A probabilistic classifier trained using features engineered from the tweet text.

Deep Learning Model:

- LSTM with GloVe Embeddings: A Long Short-Term Memory (LSTM) model built using pre-trained GloVe word embeddings. This model is designed to capture the sequential nature of text data for improved sentiment classification.

Key Features:
- Text Preprocessing:
Tokenization, handling of hashtags, @user mentions, and other tweet-specific characteristics.
Integration of pre-trained GloVe word embeddings for feature representation in the LSTM model.
- Model Training and Evaluation:
Implementation of various classifiers with careful tuning of hyperparameters.
Evaluation of model performance using macroaveraged F1 score for the positive and negative classes.
Error analysis through confusion matrices to identify and address classification challenges.

Installation and Setup:

- Visit GloVe Project and download the glove.6B.zip file.
- Unzip the file and place the extracted files in the data directory.

Run the Jupyter Notebook:
Open Sentiment_Classifier.ipynb in Jupyter Notebook or JupyterLab to explore the full implementation.

Running the Models:
Each model is implemented in a modular way, allowing you to train and evaluate them separately. Simply run the provided Jupyter Notebook, and the models will be trained and evaluated on the provided dataset.

Evaluation:
The models are evaluated using the macroaveraged F1 score for positive and negative classes. The provided evaluation script will generate these metrics along with a confusion matrix for further error analysis.

Results:
The repository includes the results of different classifiers across the multiple test sets, demonstrating their ability to generalize sentiment classification across various scenarios.

Future Work:
Possible extensions include experimenting with different neural architectures, such as Transformers, or enhancing feature engineering techniques for traditional models.
