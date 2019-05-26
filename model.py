import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline

import spacy
import pickle
from utils import spacy_tokenizer
from utils import predictors

# Loading TSV file
df_feedback = pd.read_csv ("feedback-training-data.tsv", sep="\t")
# Top 5 records
print(df_feedback.head())


# shape of dataframe
print('...............df_feedback.shape')
print(df_feedback.shape)

# View data information
print('...............df_feedback.info()')
print(df_feedback.info())

# Feedback Value count
print('...............df_feedback.Result.value_counts()')
print(df_feedback.Result.value_counts())


#******************************************creating bow vector with custom tokenizer 
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#calculate tf-idf
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

#********************spliting into training set...................
from sklearn.model_selection import train_test_split
X = df_feedback['Text'] # the features we want to analyze
ylabels = df_feedback['Result_Label'] # the labels, or answers, we want to test against
#X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2)
print('X........................')
print(X)
print(ylabels)


#********************************************Creating a Pipeline and Generating the Model .................
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                ('vectorizer', bow_vector),
                ('classifier', classifier)])

# model generation
pipe.fit(X,ylabels)


from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(ylabels, predicted))
print("Logistic Regression Precision:",metrics.precision_score(ylabels, predicted, average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(ylabels, predicted, average='micro'))


# Saving model to disk
list_pickle = open('model.pkl', 'wb')
pickle.dump(pipe, list_pickle)
list_pickle.close()

# Loading model to compare the results

list_pickle = open('model.pkl', 'rb')
model = pickle.load(list_pickle)

testDataframe = pd.DataFrame([' lab '])
print('........................................')
print(model.predict(testDataframe.iloc[0]))
