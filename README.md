# Amazon Review Sentiment

**WGU D803 · Natural Language Processing**

Reads Amazon product reviews and labels each one positive or negative. Useful when there are too many reviews to read, and star ratings don't match what people actually wrote.

## How it works

1. **Clean:** tidies the raw review text. Both the raw and cleaned files are included, so you can see what changed.
2. **Turn words into numbers:** TF-IDF gives more weight to words that separate good reviews from bad ones, and less to words that appear everywhere.
3. **Classify:** logistic regression, which is fast on text data and shows which words drive each decision.
4. **Test:** trains on 80% of reviews, tests on 20%, with a fixed seed so results repeat.
5. **Score:** accuracy plus precision, recall and F1 for each side, and a confusion matrix showing which way the mistakes go.

## Run it

```
python sentiment_model.py
```

Prints accuracy, the full classification report and the confusion matrix.

## Files

- `sentiment_model.py`: the whole pipeline
- `data/amazon_reviews_500 (1).csv`: raw reviews
- `data/amazon_reviews_cleaned (1).csv`: cleaned reviews used for training

**Built with:** Python, pandas, scikit-learn
