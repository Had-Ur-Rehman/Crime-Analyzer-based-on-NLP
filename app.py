import pandas as pd
import numpy as np
import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
import pickle
import spacy
import en_core_web_md
from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL, DATA
import werkzeug
from werkzeug.utils import secure_filename

#Scraping
from bs4 import BeautifulSoup
import requests
from datetime import timedelta, date






app = Flask(__name__)
Bootstrap(app)

# Configuration
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadstorage'
configure_uploads(app, files)

import os
import datetime
import time

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/datauploads', methods=['GET', 'POST'])
def datauploads():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = file.filename
		file.save(os.path.join('static/uploadstorage', filename))
		#EDA
		# df = pd.read_csv(os.path.join('static/uploadstorage', filename), encoding='cp1252')
		# df_table = df

		# Model Training
		# whole_News = pd.read_csv(r"whole-corpus.csv", encoding='latin-1')
		whole_News = pd.read_csv(os.path.join('static/uploadstorage', filename), encoding='cp1252')
		df_table = whole_News
		labelled_News = pd.read_csv(r"label-corpus.csv", encoding='latin-1')

		# Step - a : Remove blank rows if any.
		labelled_News['Summary'].dropna(inplace=True)
		whole_News['Summary'].dropna(inplace=True)

		# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
		labelled_News['Summary'] = [entry.lower() for entry in labelled_News['Summary']]
		whole_News['Summary'] = [entry.lower() for entry in whole_News['Summary']]

		# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
		labelled_News['Summary'] = [word_tokenize(entry) for entry in labelled_News['Summary']]
		whole_News['Summary'] = [word_tokenize(entry) for entry in whole_News['Summary']]

		# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
		# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
		tag_map = defaultdict(lambda: wn.NOUN)
		tag_map['J'] = wn.ADJ
		tag_map['V'] = wn.VERB
		tag_map['R'] = wn.ADV

		# for labelled News
		for index, entry in enumerate(labelled_News['Summary']):
			# Declaring Empty List to store the words that follow the rules for this step
			Final_words = []
			# Initializing WordNetLemmatizer()
			word_Lemmatized = WordNetLemmatizer()
			# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
			for word, tag in pos_tag(entry):
				# Below condition is to check for Stop words and consider only alphabets
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
					Final_words.append(word_Final)
			# The final processed set of words for each iteration will be stored in 'text_final'
			labelled_News.loc[index, 'text_final'] = str(Final_words)


		# for unlabelled News

		for index, entry in enumerate(whole_News['Summary']):
			# Declaring Empty List to store the words that follow the rules for this step
			Final_words = []
			# Initializing WordNetLemmatizer()
			word_Lemmatized = WordNetLemmatizer()
			# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
			for word, tag in pos_tag(entry):
				# Below condition is to check for Stop words and consider only alphabets
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
					Final_words.append(word_Final)
			# The final processed set of words for each iteration will be stored in 'text_final'
			whole_News.loc[index, 'summary_final'] = str(Final_words)

		# spliting labelled News for model training
		# X is News, Y is Label

		Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(labelled_News['text_final'],
																			labelled_News['label'], test_size=0.3)

		Encoder = LabelEncoder()
		Train_Y = Encoder.fit_transform(Train_Y)
		Test_Y = Encoder.fit_transform(Test_Y)

		testing_set = dict(zip(Test_X, Test_Y))

		Tfidf_vect = TfidfVectorizer(max_features=5000)
		Tfidf_vect.fit(labelled_News['text_final'])
		Train_X_Tfidf = Tfidf_vect.transform(Train_X)
		Test_X_Tfidf = Tfidf_vect.transform(Test_X)

		whole_News_Tfidf = Tfidf_vect.transform(whole_News['summary_final'])
		#
		# Tfidf_vect = TfidfVectorizer(max_features=5000)
		# Tfidf_vect.fit(labelled_News['text_final'])
		# whole_News_Tfidf = Tfidf_vect.transform(whole_News['summary_final'])

		SVM_f = open('crime_svm.pickle', 'rb')
		SVM = pickle.load(SVM_f)
		SVM_f.close()
		# Classifier - Algorithm - SVM
		# fit the training dataset on the classifier
		# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
		# save_svm = SVM.fit(Train_X_Tfidf, Train_Y)
		# # predict the labels on validation dataset
		# data = open("crime_svm.pickle", "wb")
		# pickle.dump(save_svm, data)
		# data.close()

		# predictions_SVM = SVM.predict(Test_X_Tfidf)

		whole_News_predictions = SVM.predict(whole_News_Tfidf)
		whole_News_Document = "whole-corpus.csv"
		fields = []
		rows = []
		i = 0  # for iteration through labels

		with open(whole_News_Document, 'r') as csvfile:
		# creating a csv reader object
			csvreader = csv.reader(csvfile)
			fields = next(csvreader)

			# extracting field names through first row

			# extracting each data row one by one
			for row in csvreader:
				if whole_News_predictions[i] == 0:
					rows.append(row)

				i = i + 1
		#Crime News df1
		df1 = pd.DataFrame(rows, columns=['Headlines', 'Summary', 'Time'])
		#Named Entities
		snlp = en_core_web_md.load()

		entities = ['GPE', 'LOC', 'PERSON', 'ORG', 'TIME']

		df1['ner_text'] = df1['Summary'].astype(str).apply(lambda x: list(snlp(x).ents))

		# this loop labels entities which are bulit in spacy
		for i in range(len(entities)):
			df1[entities[i]] = df1['Summary'].astype(str).apply(
				lambda x: [t.text for t in snlp(x).ents if t.label_ == entities[i]])

		df1['GPE'] = df1['Summary'].astype(str).apply(lambda x: [t.text for t in snlp(x).ents if t.label_ == 'GPE'])

		# below lines call the Crime_Entity_Model and labels Crime Entities for the crime news
		nlp = spacy.load('Crime-Model1')

		df1['Type of Crime'] = df1['Summary'].astype(str).apply(lambda x: list({t.label_ for t in nlp(x).ents}))


		# del df['ner_text']
		# del df['Headlines']
		df_table = df1
	# return render_template('details.html', filename=filename, df_table=df1)
	return df1.to_html()


@app.route('/Scrape', methods=['POST'])
def Scrape():
	date1 = request.form['text1']
	year1 = int(date1[0:4])

	m1 = int(date1[5:7])
	d1 = int(date1[8:10])
	# print(year1)
	# print(m1)
	# print(d1)
	date2 = request.form['text2']
	year2 = int(date2[0:4])
	# m2 = int(date2[0:2])
	# d2 = int(date2[3:5])
	# year2 = int(date2[6:10])
	m2 = int(date2[5:7])
	d2 = int(date2[8:10])

	csv_file = open('Scrape.csv', 'w', newline='')

	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['Headlines', 'Summary', 'Time'])

	def daterange(start_date, end_date):
		for n in range(int((end_date - start_date).days)):
			yield start_date + timedelta(n)

	start_date = date(year1, m1, d1)
	end_date = date(year2, m2, d2)

	sumry = [] #list of Summary
	t = [] #list of time
	head = []
	for single_date in daterange(start_date, end_date):
		try:
			time = single_date.strftime("%Y-%m-%d")
			link = f'https://www.dawn.com/newspaper/national/{time}'

			source = requests.get(link).text

			soup = BeautifulSoup(source, 'html.parser')
			article = soup.find('article', class_='story story--large')
			headline = article.h2.a.text
			# print(headline)

			summary = article.find('div', class_='story__excerpt text-4').text
			# print(summary)
			# time = article.find('span', class_='timestamp--time timeago')['title']
			# time = time.split('T')
			# time = time[0]
			# print(time)
			sumry.append(summary)
			t.append(time)
			head.append(headline)
			csv_writer.writerow([headline, summary, time])

			for article1 in soup.find_all('article', class_='story story--small'):
				headline1 = article1.h2.a.text
				# print(headline1)

				summary1 = article1.find('div', class_='story__excerpt text-4').text
				# print(summary1)
				#     time1 = article1.find('span', class_='timestamp--time timeago')['title']
				#     time1 = time1.split('T')
				#     time1 = time1[0]
				# print(time)
				sumry.append(summary1)
				t.append(time)
				head.append(headline1)
				# print()
				csv_writer.writerow([headline1, summary1, time])
		except Exception as e:
			pass

	csv_file.close()

	whole_News = pd.DataFrame(list(zip(head,sumry, t)), columns =['Headline', 'Summary', 'Time'])

	df_table = whole_News
	labelled_News = pd.read_csv(r"label-corpus.csv", encoding='latin-1')

	# Step - a : Remove blank rows if any.
	labelled_News['Summary'].dropna(inplace=True)
	whole_News['Summary'].dropna(inplace=True)

	# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
	labelled_News['Summary'] = [entry.lower() for entry in labelled_News['Summary']]
	whole_News['Summary'] = [entry.lower() for entry in whole_News['Summary']]

	# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
	labelled_News['Summary'] = [word_tokenize(entry) for entry in labelled_News['Summary']]
	whole_News['Summary'] = [word_tokenize(entry) for entry in whole_News['Summary']]

	# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
	# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
	tag_map = defaultdict(lambda: wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV

	# for labelled News
	for index, entry in enumerate(labelled_News['Summary']):
		# Declaring Empty List to store the words that follow the rules for this step
		Final_words = []
		# Initializing WordNetLemmatizer()
		word_Lemmatized = WordNetLemmatizer()
		# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
		for word, tag in pos_tag(entry):
			# Below condition is to check for Stop words and consider only alphabets
			if word not in stopwords.words('english') and word.isalpha():
				word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
				Final_words.append(word_Final)
		# The final processed set of words for each iteration will be stored in 'text_final'
		labelled_News.loc[index, 'text_final'] = str(Final_words)

	# for unlabelled News

	for index, entry in enumerate(whole_News['Summary']):
		# Declaring Empty List to store the words that follow the rules for this step
		Final_words = []
		# Initializing WordNetLemmatizer()
		word_Lemmatized = WordNetLemmatizer()
		# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
		for word, tag in pos_tag(entry):
			# Below condition is to check for Stop words and consider only alphabets
			if word not in stopwords.words('english') and word.isalpha():
				word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
				Final_words.append(word_Final)
		# The final processed set of words for each iteration will be stored in 'text_final'
		whole_News.loc[index, 'summary_final'] = str(Final_words)

	# spliting labelled News for model training
	# X is News, Y is Label

	Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(labelled_News['text_final'],
																		labelled_News['label'], test_size=0.3)

	Encoder = LabelEncoder()
	Train_Y = Encoder.fit_transform(Train_Y)
	Test_Y = Encoder.fit_transform(Test_Y)

	testing_set = dict(zip(Test_X, Test_Y))

	Tfidf_vect = TfidfVectorizer(max_features=5000)
	Tfidf_vect.fit(labelled_News['text_final'])
	Train_X_Tfidf = Tfidf_vect.transform(Train_X)
	Test_X_Tfidf = Tfidf_vect.transform(Test_X)

	whole_News_Tfidf = Tfidf_vect.transform(whole_News['summary_final'])
	#
	# Tfidf_vect = TfidfVectorizer(max_features=5000)
	# Tfidf_vect.fit(labelled_News['text_final'])
	# whole_News_Tfidf = Tfidf_vect.transform(whole_News['summary_final'])

	SVM_f = open('crime_svm.pickle', 'rb')
	SVM = pickle.load(SVM_f)
	SVM_f.close()
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	# save_svm = SVM.fit(Train_X_Tfidf, Train_Y)
	# # predict the labels on validation dataset
	# data = open("crime_svm.pickle", "wb")
	# pickle.dump(save_svm, data)
	# data.close()

	# predictions_SVM = SVM.predict(Test_X_Tfidf)

	whole_News_predictions = SVM.predict(whole_News_Tfidf)
	whole_News_Document = "Scrape.csv"
	fields = []
	rows = []
	i = 0  # for iteration through labels

	with open(whole_News_Document, 'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile)
		fields = next(csvreader)

		# extracting field names through first row

		# extracting each data row one by one
		for row in csvreader:
			if whole_News_predictions[i] == 0:
				rows.append(row)

			i = i + 1
	# Crime News df1
	df1 = pd.DataFrame(rows, columns=['Headlines', 'Summary', 'Time'])
	# Named Entities
	snlp = en_core_web_md.load()

	entities = ['GPE', 'LOC', 'PERSON', 'ORG', 'TIME']

	df1['ner_text'] = df1['Summary'].astype(str).apply(lambda x: list(snlp(x).ents))

	# this loop labels entities which are bulit in spacy
	for i in range(len(entities)):
		df1[entities[i]] = df1['Summary'].astype(str).apply(
			lambda x: [t.text for t in snlp(x).ents if t.label_ == entities[i]])

	df1['GPE'] = df1['Summary'].astype(str).apply(lambda x: [t.text for t in snlp(x).ents if t.label_ == 'GPE'])

	# below lines call the Crime_Entity_Model and labels Crime Entities for the crime news
	nlp = spacy.load('Crime-Model1')

	df1['Type of Crime'] = df1['Summary'].astype(str).apply(lambda x: list({t.label_ for t in nlp(x).ents}))

	# del df['ner_text']
	# del df['Headlines']
	df_table = df1

	return df1.to_html()


if __name__ == "__main__":
	app.run()  # Flask
