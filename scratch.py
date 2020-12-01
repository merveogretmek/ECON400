import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# read data from csv
data = pd.read_csv("amazon_data.csv")

# remove the row if star_rating = 3 (cannot interpret 3 as positive or negative)
data = data[data['star_rating'] != 3]

# create a new column "feedback sentiment"
# if star_rating is above 3, sentiment is positive ; if star_rating is below 3, sentiment is negative
data['feedback sentiment'] = data['star_rating'].apply(lambda rating : "Positive" if rating > 3 else "Negative")

# choose only two columns from the data
# review_body = text of the feedback, feedback sentiment = positive/negative
data = data[['review_body','feedback sentiment']]

# 80% of the observations will be used for training (240000/300000)
train = data.head(240000)

# 20% of the observations will be used for training (60000/300000)
test = data.tail(60000)

# regular expression '\b\w+\b' will capture the words from the text by escaping whitespace
# create a vocabulary pool from the observed text
tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')


# create train_matrix
# astype(U) : convert values to Unicode (feedback text might contain many different special characters)
train_matrix = tokenizer.fit_transform(train['review_body'].values.astype('U'))

# first 5 line from printing train_matrix :
#  (0, 1383)	1
#  (0, 34257)	1
#  (0, 43930)	1
#  (0, 31836)	3
#  (0, 64184)	1

# Observe the first row:
# 0 is the sentence index from tokenizer.vocabulary_
# 1383 the word index from tokenizer.vocabulary_
# 1 is the number of times the word(with index number = 1383) appears in this feedback text

#create test_matrix
test_matrix = tokenizer.transform(test['review_body'].values.astype('U'))

# define a Logistic Regression Model: classifier
logit = LogisticRegression()

# fit the model with given training data
logit.fit(train_matrix,train['feedback sentiment'])

# predict the sentiment for observations in testing data
prediction = logit.predict(test_matrix)

# print confusion matrix to compare the predicted and actual results and
print(confusion_matrix(prediction,test['feedback sentiment']))

# classsification report: text report showing the main classification metrics.
print(classification_report(prediction,test['feedback sentiment']))





