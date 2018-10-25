# Text Classifiation using NLP
import sys

# Importing the libraries
import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

# Unpickling dataset
X_in = open('X.pickle','rb')
y_in = open('y.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


# Using our classifier
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
    

sample = sys.argv[1]


sample = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", sample)
sample = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", sample)
sample = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", sample)
sample = sample.lower()
sample = re.sub(r"that's","that is",sample)
sample = re.sub(r"there's","there is",sample)
sample = re.sub(r"what's","what is",sample)
sample = re.sub(r"where's","where is",sample)
sample = re.sub(r"it's","it is",sample)
sample = re.sub(r"who's","who is",sample)
sample = re.sub(r"i'm","i am",sample)
sample = re.sub(r"she's","she is",sample)
sample = re.sub(r"he's","he is",sample)
sample = re.sub(r"they're","they are",sample)
sample = re.sub(r"who're","who are",sample)
sample = re.sub(r"ain't","am not",sample)
sample = re.sub(r"wouldn't","would not",sample)
sample = re.sub(r"shouldn't","should not",sample)
sample = re.sub(r"can't","can not",sample)
sample = re.sub(r"couldn't","could not",sample)
sample = re.sub(r"won't","will not",sample)
sample = re.sub(r"\W"," ",sample)
sample = re.sub(r"\d"," ",sample)
sample = re.sub(r"\s+[a-z]\s+"," ",sample)
sample = re.sub(r"\s+[a-z]$"," ",sample)
sample = re.sub(r"^[a-z]\s+"," ",sample)
sample = re.sub(r"\s+"," ",sample)

sample = tfidf.transform([sample]).toarray()
sentiment = clf.predict(sample)

if sentiment[0] == 1:
	print("positive")
else:
	print("Negative")