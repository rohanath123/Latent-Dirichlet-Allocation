#STABLE VERSION OF LDA USING GENSIM IMPLEMENTATION, DO NOT ALTER

import re 
import numpy as np 
import pandas as pd 
from pprint import pprint

import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt 

#PREPARING STOPWORDS
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#IMPORTING NEWSGROUPS DATASET
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
data = df.content.values.tolist()

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

data = [re.sub("\'", "", sent) for sent in data]

def sent_to_words(sentences):
	for sent in sentences:
		yield(gensim.utils.simple_preprocess(str(sent), deacc = True))

data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 100)
trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
	return([word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts)

def make_bigrams(texts):
	return([bigram_mod[doc] for doc in texts])

def make_triagrams(texts):
	return([trigram_mod[doc] for doc in texts])

def lemmatization(texts, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent))
		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
	return texts_out

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load('en_core_web_sm', diasable = ['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
	id2word = id2word, 
	num_topics = 20, 
	random_state = 100, 
	update_every = 1, 
	chunksize = 100, 
	passes = 10, 
	alpha = 'auto', 
	per_word_topics = True)


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]