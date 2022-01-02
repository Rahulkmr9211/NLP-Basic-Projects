# -*- coding: utf-8 -*-
"""
Author = Rahul Kumar
Page = SMS Spam Classifier
"""

import pandas as pd

message = pd.read_csv('SMSSpamCollection',sep='\t', names=['label','message'])

#check data balance
hams = 0
spams = 0
for i in message['label']:
    if i == 'ham':
        hams += 1
    elif i == 'spam':
        spams +=1

print('Hams % :', round(hams/len(message)*100,2))
print('spams % :', round(spams/len(message)*100,2))

#Data Cleaning and Preprocessing
import re   #
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(len(message)):
    review = re.sub('[^a-zA-Z]',' ',message['message'][i]) #except words, everything gets removed
    review = review.lower() #message words lower case
    review = review.split() #split where ' ' to get list of words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(message['label'])
y = y.iloc[:,1].values

#Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

#for accuracies
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = round(accuracy_score(y_test, y_pred),2)
