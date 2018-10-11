"""IMPORTS"""
import pickle
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
# import re
# import pprint
from nltk import word_tokenize
""""
preprocess.py will open any type of csv file and parse through
each line and preprocess by removing stopwords, punctuation, etc...

Author: Jessy Rakesh Armstrong
Email: jessyarmstrong761@gmail.com
github: JessyRakesh-Armstrong
"""


def normalize(comment):
    """This is where all the normalizing happens"""
    norm_str = tokenize_comment(comment)
    return norm_str


def tokenize_comment(raw_str):
    """
    Uses the nltk commment to tokenize and return that as a list
    Itype: String
    Rtype: String
    """
    tok_comments = word_tokenize(raw_str)
    return tok_comments


def read_csv(csv1, csv2):
    """
    Itype: csv files
    Rtype: tuple of pandas data frames
    """
    train_df = pd.read_csv(csv1)
    test_df = pd.read_csv(csv2)
    return train_df, test_df


def create_tdidf_vec(train, test):
    """
    Itype: Pandas.df, Pandas.df
    Rtype: a tuple of transformed vectors
    """
    vec = TfidfVectorizer()
    train_vectors = vec.fit_transform(train['comment_text'].values.astype('U'))
    test_vectors = vec.transform(test['comment_text'].values.astype('U'))
    return train_vectors, test_vectors


def get_mdl(vec, y):
    """

    """

    # Helper function for get_mdl()
    def pr(vec, y_i, y):
        p = vec[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum()+1)

    y = y.values
    ratio = np.log(pr(vec, 1, y) / pr(vec, 0, y))
    model = LogisticRegression(C=4, dual=True)
    nb = vec.multiply(ratio)
    return model .fit(nb, y), ratio


def train_model():
    """
    Actual source calling all functions to iterate through the csv file
    and train on the model using the data
    """
    labels = [
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
        ]

    train_df, test_df = read_csv('data/train.csv', 'data/test.csv')
    train_vec, test_vec = create_tdidf_vec(train_df, test_df)

    saved_models = []
    preds = np.zeros((len(test_df), len(labels)))

    for i, label in enumerate(labels):
        print('fitting', label)
        m, r = get_mdl(train_vec, train_df[label])
        s = pickle.dumps(m)
        saved_models.append(s)
        preds[:, i] = m.predict_proba(test_vec.multiply(r))[:, 1]
    return preds


def predict_csv(predictions):
    """
    Predict csv test file by using train model
    """
    train_df, test_df = read_csv('data/train.csv', 'data/test.csv')
    labels = [
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
    ]
    # predicted_df = pd.DataFrame(data=predictions, columns=labels)
    classified_preds = np.zeros(len(test_df), len(labels))
    for row in range(len(classified_preds)):
        for col in range(len(labels)):
            if classified_preds[row][col] > .5:
                classified_preds[row][col] = 1
            else:
                classified_preds[row][col] = 0
    predicted_df = pd.DataFrame(data=classified_preds, columns=labels)
    return predicted_df


def main():

    tr = train_model()
    prediction = predict_csv(tr)
    prediction.to_csv(
        "classfied_comments.csv",
        encoding='utf-8',
        quoting=csv.QUOTE_NONNUMERIC,
        index=False
    )


if __name__ == "__main__":
    main()
