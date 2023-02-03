
# Basic Python Libraries
import re
import requests
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import xml.etree.ElementTree as ET

# Natural Language Processing
import nltk
from nltk.corpus import stopwords ## A collection of words that don’t provide any meaning to a sentence.
from nltk.stem import WordNetLemmatizer

from happytransformer import HappyTextClassification 

def determine_sentiment(text: str):
    happy_tc = HappyTextClassification("BERT", "ProsusAI/finbert", num_labels=3)
    result = happy_tc.classify_text(text) ## Determine the sentiment of text
    label = result.label
    score = result.score
    return label

def main():
    x = requests.get('http://feeds.feedburner.com/zerohedge/feed')

    # print(x.text) # GET XML FROM ABOVE URL

    tree = ET.ElementTree(ET.fromstring(x.text)) # CREATE ELEMENT TREE OF XML

    root = tree.getroot() # RSS TAG

    for item in root.iter('item'):
        title = item.find('title').text
        print(determine_sentiment(title))


if __name__ == "__main__":
    main()
