**Overview**

This repository contains my completed submission for the WGU course **D803: Natural Language Processing**.

The project builds a sentiment analysis classifier that determines whether an Amazon product review expresses positive or negative sentiment. Unstructured review text is cleaned, converted into a numerical feature representation using TF-IDF, and classified with logistic regression. Model quality is assessed using precision, recall, F1, and a confusion matrix rather than accuracy alone.

**Scenario**

A retailer receives large volumes of free-text product reviews. Reading them manually does not scale, and raw star ratings do not always reflect the sentiment expressed in the written feedback.

The objective is to build a model that reads review text directly and classifies its sentiment, enabling automated monitoring of customer opinion at volume.

**Project Objectives**

-Prepare and clean a raw corpus of Amazon product reviews for modeling

-Convert unstructured text into numerical feature vectors suitable for a classifier

-Train a supervised sentiment classification model

-Evaluate the model using metrics appropriate to classification, not just overall accuracy

-Interpret the confusion matrix to understand which class the model struggles with

**Technical Implementation**

**Text Preprocessing**
The raw review corpus is cleaned into a normalized text column prior to modeling. Both the original and cleaned datasets are retained in the repository so the preprocessing step is transparent and reviewable.

**Feature Extraction: TF-IDF**
Reviews are vectorized using term frequency inverse document frequency. TF-IDF was selected over raw count vectorization because it down-weights terms that appear across nearly every review and gives more signal to terms that distinguish one sentiment class from another.

**Model: Logistic Regression**
A logistic regression classifier is trained on the vectorized corpus with an increased iteration limit to ensure convergence on the high-dimensional sparse feature matrix. Logistic regression is well suited to sparse text features, trains quickly, and produces interpretable coefficients that map back to individual terms.

**Train/Test Split**
An 80/20 split with a fixed random seed keeps evaluation reproducible across runs.

**Evaluation**
The model is scored on overall accuracy and a full classification report covering precision, recall, and F1 for each sentiment class. A confusion matrix is generated to expose the specific direction of misclassification.

**Skills Demonstrated**

-Natural language processing and text preprocessing

-TF-IDF vectorization and feature extraction from unstructured text

-Supervised classification with logistic regression

-Train/test methodology and reproducible evaluation

-Classification metrics: precision, recall, F1, and confusion matrix analysis

-Working with high-dimensional sparse feature representations

-Applied sentiment analysis on real-world review data

**How to Run**

```
python sentiment_model.py
```

The script expects the cleaned review dataset in the data directory and prints accuracy, a full classification report, and the confusion matrix to the console.

**Repository Contents**

-`sentiment_model.py`: Vectorization, training, and evaluation pipeline

-`data/amazon_reviews_500 (1).csv`: Raw Amazon review dataset

-`data/amazon_reviews_cleaned (1).csv`: Preprocessed dataset used for modeling
