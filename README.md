# Sentiment Analysis on IMDB Reviews using LSTM

This repository contains a project for performing sentiment analysis on IMDB movie reviews using an LSTM (Long Short-Term Memory) neural network. The project focuses on understanding and predicting whether a given movie review is positive or negative based on its textual content.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Sentiment analysis is a critical task in Natural Language Processing (NLP), where the goal is to determine the sentiment or emotional tone behind a text. This project applies an LSTM model, a type of recurrent neural network (RNN) known for its ability to capture long-term dependencies, to classify movie reviews from the IMDB dataset as either positive or negative.

The focus of this project is to preprocess textual data, tokenize reviews, pad sequences, and train an LSTM model for effective sentiment classification.

## Dataset

The dataset used in this project is the IMDB dataset of 50,000 movie reviews, which includes both positive and negative sentiments.

Source: The dataset is publicly available on Kaggle.

- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install pandas tensorflow scikit-learn kaggle


```

Requirements
Python 3.x
TensorFlow
Pandas
Scikit-learn

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Sentiment-Analysis-on-IMDB-Reviews-using-LSTM.git
```

2. Navigate to the project directory:
   cd Sentiment-Analysis-on-IMDB-Reviews-using-LSTM

3. Open and run the Jupyter Notebook:
   jupyter notebook IMDB_Reviews_Sentiment_Analysis.ipynb

## Model

The model used in this project is an LSTM neural network implemented using TensorFlow and Keras.

### Architecture

1. Embedding Layer: Converts input text into dense vectors of fixed size for processing.
2. LSTM Layer: Captures the sequential nature of text data and long-term dependencies.
3. Dense Layer: Final fully connected layer with a sigmoid activation function for binary classification.

### Training

1. Loss Function: Binary Crossentropy for binary classification.
2. Optimizer: Adam optimizer for adaptive learning rate optimization.
3. Metrics: Accuracy is used to evaluate the model's performance.
