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


def normalize(comment):
    """This is where all the normalizing happens"""
    norm_str = tokenize_comment(comment)
    return norm_str


def tokenize_comment(raw_str):
    """
    Uses the nltk commment to tokenize and return that as a list
    """
    tok_comments = word_tokenize(raw_str)
    return tok_comments


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
