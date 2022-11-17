import nltk
nltk.download('punkt')
nltk.download('words')
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer

import re
import os
import json
import datetime
import csv
import numpy as np
import time

def get_raw_training_data(csv_name):
    training_data = []
    with open(csv_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            # print(row)

            dict = {}
            dict['person'] = row[0]
            dict['sentence'] = row[1]
            training_data.append(dict)
            print('\n')
            # print('-----')
    # print(training_data)
    return training_data

    
def preprocess_words(words, stemmer):
    token_set = set()
    token_list = [stemmer.stem(word) for word in words]
    for token in token_list:
        token_set.add(token)

    return list(token_set)

def organize_raw_training_data(raw_training_data, stemmer):
    words = []
    documents = []
    classes = []
    
    for dict in raw_training_data:
        sentence = dict['sentence']
        person = dict['person']
        py_opstr = re.sub(r'[^\w\s]','', sentence)
        tokenize = word_tokenize(py_opstr)
        # print("tokens given to preproccess")
        # print(tokenize)
        # print('--------')
        sent_words = list(preprocess_words(tokenize, stemmer))
        words.append(sent_words)
        documents.append((sent_words, person))
        if person not in classes: 
            classes.append(person)

    return words, classes, documents
       
def create_training_data(words, classes, documents, stemmer): 
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    training_data = []
    output = []

    for tuple in documents:
        curr_data = []
        for word in tuple[0]:
            if word in english_vocab:
                curr_data.append(1)
            else:
                curr_data.append(0)
        training_data.append(curr_data)
        
        output_data = [0] * len(classes)
        speaker = tuple[1]
        speaker_index = classes.index(speaker)
        output_data[speaker_index] = 1
        output.append(output_data)


    # training_data = [word for word in word_list if word in english_vocab]

    return training_data, output



def main():
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('dialogue_data.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(words, classes, documents, stemmer)
    # print(documents)
    
    print(training_data)
    print(output)
if __name__ == "__main__":
    main()




