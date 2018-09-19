"""IMPORTS"""
import pandas as pd
import numpy as np
import nltk
import re
import pprint
from nltk import word_tokenize
""""
preprocess.py will open any type of csv file and parse through
each line and preprocess by removing stopwords, punctuation, etc...

Author: Jessy Rakesh Armstrong
Email: jessyarmstrong761@gmail.com
github: JessyRakesh-Armstrong
"""


def tokenize_comment(comment):
    """
    Uses the nltk commment to tokenize and return that as a list
    """
    tok_comment = word_tokenize(comment)
    return tok_comment


def main():
    """
    Actual source calling all functions to iterate through the csv file
    """
    used_cols = [
        'comment_text',
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
        ]

    train_df = pd.read_csv('data/train.csv',
                           dtype={'comment_text': np.str},
                           usecols=used_cols)
    print(tokenize_comment(train_df['comment_text'][0]))


if __name__ == "__main__":
    """
    Main function which will iterate the preprocessing routine for each
    all the lines. I guess it also looks pretty
    """
    main()
