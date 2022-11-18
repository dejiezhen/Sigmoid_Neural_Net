"""
Bjorn Ludwig, and Dejie Zhen
CSCI 3725
PQ4: Social Networks?
11/17/22
The dialogue_classifier.py file helps process the text data for the neural 
network. We used NLTK to help us understand the text better with tokenization 
and stemming. We also created a bag of words training data for the words and 
output data for the classes to train the classifying network.

Dependencies: nltk, csv, re
"""

import nltk
nltk.download('punkt')
nltk.download('words')
from nltk.tokenize import word_tokenize
import re
import csv

def get_raw_training_data(csv_name):
    """
    Parses the csv file to create an array that contains our raw data.
    """
    training_data = []
    with open(csv_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            dict = {}
            sentence_raw = row[1]
            sentence = sentence_raw.replace('...', ' ')
            dict['person'] = row[0].replace('"', '')
            dict['sentence'] = sentence
            training_data.append(dict)

    return training_data

    
def preprocess_words(words, stemmer):
    """
    For a given list of words, return their unique stems.
    """
    token_set = set()
    token_list = [stemmer.stem(word) for word in words]
    
    for token in token_list:
        token_set.add(token)

    return list(token_set)

def organize_raw_training_data(raw_training_data, stemmer):
    """
    Takes in raw training data and returns a list of all its unique words, 
    the speakers, and an array of tuples of sentences with associated speaker. 
    """
    words = set()
    documents = []
    classes = []
    
    for dict in raw_training_data:
        sentence = dict['sentence']
        person = dict['person']
        py_opstr = re.sub(r'[^\w\s]','', sentence)
        tokenize = word_tokenize(py_opstr)
        sent_words = list(preprocess_words(tokenize, stemmer))       
    
        for word in sent_words:
            words.add(word)
        documents.append((sent_words, person))

        if person not in classes: 
            classes.append(person)

    return list(words), classes, documents
       
def create_training_data(words, classes, documents, stemmer): 
    """
    Create bag of words training data and output using words, documents, 
    and classes.
    """
    training_data = []
    output = []

    for tuple in documents:
        curr_data = [0] * len(words)
        sentence = tuple[0]

        for i, word in enumerate(words):
            if word in sentence:
                curr_data[i] = 1 
        training_data.append(curr_data)
        
        output_data = [0] * len(classes)
        speaker = tuple[1]
        speaker_index = classes.index(speaker)
        output_data[speaker_index] = 1
        output.append(output_data)

    return training_data, output
