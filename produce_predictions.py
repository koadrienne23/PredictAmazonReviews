import pandas as pd
import numpy as np
import nltk
from ast import literal_eval
from textblob import TextBlob
from textblob import Word
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# %config IPCompleter.greedy=True

# Read in the data.
train_raw = pd.read_csv('./train.csv')
test_raw = pd.read_csv('./test.csv')

# Process the data.
train_raw = train_raw.dropna(
    axis='index',
    how='any',
    subset=['helpful', 'reviewText', 'summary', 'overall'])
train_raw = train_raw.reset_index(drop=True)
test_raw = test_raw.dropna(
    axis='index',
    how='any',
    subset=['helpful', 'reviewText'])
test_raw = test_raw.reset_index(drop=True)

# -- Prepare helpful for aggregation.
convert_helpful = lambda x: 0 if literal_eval(x)[1] == 0 else literal_eval(x)[0] / literal_eval(x)[1]

train_raw['helpful'] = train_raw['helpful'].apply(convert_helpful)
test_raw['helpful'] = test_raw['helpful'].apply(convert_helpful)

# -- Aggregate and organize data.
convert_overall = lambda y: 1 if np.mean(y) > 4.5 else 0

training_df = train_raw.groupby('amazon-id').agg({
    'helpful': lambda z: np.mean(z),
    'reviewText': ' '.join,
    'summary': ' '.join,
    'overall': convert_overall})
testing_df = test_raw.groupby('amazon-id').agg({
    'helpful': lambda w: np.mean(w),
    'reviewText': ' '.join,
    'summary': ' '.join})

# -- Insert duplicate columns for sentiment analysis.
training_df.insert(
    2,
    "pol_rt",
    training_df[['reviewText']].copy(),
    True)
training_df.insert(
    4,
    "pol_summary",
    training_df[['summary']].copy(),
    True)
testing_df.insert(
    2,
    "pol_rt",
    testing_df[['reviewText']].copy(),
    True)
testing_df.insert(
    4,
    "pol_summary",
    testing_df[['summary']].copy(),
    True)

# -- Sentiment analysis using TextBlob.
training_df['pol_rt'] = training_df['pol_rt'].apply(lambda x: TextBlob(x).sentiment.polarity)
training_df['pol_summary'] = training_df['pol_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
testing_df['pol_rt'] = testing_df['pol_rt'].apply(lambda x: TextBlob(x).sentiment.polarity)
testing_df['pol_summary'] = testing_df['pol_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)


# -- TFIDF using TFIDFVectorizer.
def lemmatize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for token in tokens:
        stems.append(Word(token).lemmatize())
    return stems


tfidf_transformer = ColumnTransformer(
    transformers=[
        (
            'vect_rt',
            TfidfVectorizer(
                tokenizer=lemmatize),
            'reviewText'),
        (
            'vect_summary',
            TfidfVectorizer(
                tokenizer=lemmatize),
            'summary')
    ],
    remainder='passthrough'
)

# The model.
lr_pipeline = Pipeline([
    ('tfidf', tfidf_transformer),
    ('lr', LogisticRegression(
        C=1.5,
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000))
])

# Fit the model.
# X_train = training_df[[
#         'helpful',
#         'reviewText',
#         'pol_rt',
#         'summary',
#         'pol_summary']]
# y_train = training_df['overall']
# X_test = testing_df
#
# lr_pipeline.fit(X_train, y_train)

# -- Produce 10-fold weighted-F1 score for the model.
# wf1_score = cross_val_score(
#     lr_pipeline,
#     training_df[[
#         'helpful',
#         'reviewText',
#         'pol_rt',
#         'summary',
#         'pol_summary']],
#     training_df[['overall']],
#     scoring='f1_weighted',
#     cv=10,
#     n_jobs=-1,
#     verbose=10
# )
# print("WF1: ", wf1_score.mean())
# print("Set: ", wf1_score)

# The following performs a 10-Fold CV and calculates the weighted recall and precision scores
recall_score = cross_val_score(lr_pipeline, training_df.drop('overall', axis=1), training_df[['overall']].values.ravel(), scoring='recall_weighted', cv=10)
# precision_score = cross_val_score(lr_pipeline, training_df.drop('overall', axis=1), training_df[['overall']].values.ravel(), scoring='precision_weighted', cv=10)

print("Recall: ", recall_score.mean())
# print("Precision: ", precision_score.mean())


# Sets x_train and y_train to cover all of the training dataset
x_train, y_train = training_df.drop('overall', axis=1), training_df['overall']
x_test = testing_df

# Fit model
lr_pipeline.fit(x_train, y_train)

# -- Generate and output predictions.
preds = lr_pipeline.predict(x_test)
output = pd.DataFrame({'amazon-id': x_test.index, 'Awesome': preds})
output.to_csv('./ExtraCreditPredictions.csv')


# OTHER CODE:

# -- Alternative scoring method using StratfiedShuffleSplit
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import f1_score

# s = StratifiedShuffleSplit(n_splits=10, random_state=7)
# X = training_df.drop('overall', axis=1)
# y = training_df['overall']

# scores = []
# for train, test, in s.split(X, y):
#     X_train, X_test = X.iloc[train], X.iloc[test]
#     y_train, y_test = y.iloc[train], y.iloc[test]
#     xgbc_pipeline.fit(X_train, y_train)
#     pred = xgbc_pipeline.predict(X_test)

#     score = f1_score(y_test, pred, average='weighted')
#     scores.append(score)
#     print(score)

# print(sum(scores)/len(scores))

# -- Additional model testing for Deliverable #4.
# Testing additional models for Deliverable #4.
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report


# models = [
#     ('AB', AdaBoostClassifier()),
#     ('XGB', XGBClassifier()),
#     ('LS', LinearSVC()),
#     ('LR', LogisticRegression()),
#     ('RFC', RandomForestClassifier())
# ]

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     training_df.drop('overall', axis=1),
#     training_df['overall'],
#     test_size=0.2,
#     stratify=training_df['overall'])

# for name, model in models:
#     clf_pipeline = Pipeline([
#         ('tfidf', tfidf_transformer),
#         ('clf', model)
#     ])

#     clf_pipeline.fit(X_train, y_train)
#     pred = clf_pipeline.predict(X_test)
#     print(name, ": ")
#     print(classification_report(y_test, pred))












# # COSC74 Project - Deliverable #4
# # Adrienne Ko, Jakob Kim, Kelly Song
#
# # The relevant libraries.
# import pandas as pd
# import numpy as np
# from ast import literal_eval
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score
# from nltk.stem import WordNetLemmatizer
# from nltk import word_tokenize
# from textblob import TextBlob
# # %config IPCompleter.greedy=True
#
#
# # Loads the data into a Pandas dataframe object
# train_r = pd.read_csv('./train.csv')
# test_r = pd.read_csv('./test.csv')
#
#
# # Drop null reviews
# train_r = train_r.dropna(axis='index', how='any')
# train_r = train_r.reset_index(drop=True)
#
#
# # Converts the helpful feature into a decimal numbers
# train_r['helpful'] = train_r['helpful'].apply(lambda x: 0 if literal_eval(x)[1]==0 else literal_eval(x)[0]/literal_eval(x)[1])
# test_r['helpful'] = test_r['helpful'].apply(lambda x: 0 if literal_eval(x)[1]== 0 else literal_eval(x)[0]/literal_eval(x)[1])
#
#
# # Drop all unhelpful (<0.9) reviews from the training dataset
# indexes_to_drop = []
#
# for i in range(len(train_r['helpful'])):
#     if train_r['helpful'][i] == np.nan or train_r['helpful'][i] < 0.9:
#         indexes_to_drop.append(i)
#
# train_r.drop(train_r.index[indexes_to_drop], inplace=True)
# train_r = train_r[train_r['helpful'].notna()]
#
#
# # Define functions to help aggregate columns by product
# convert_overall = lambda x: 1 if np.mean(x) > 4.5 else 0
# agg_review = lambda y: "".join(str(y))
#
# # Aggregate select features by product
# training_df = train_r.groupby('amazon-id').agg({
#     'helpful': lambda z: np.mean(z),
#     'reviewText': ' '.join,
#     'summary': ' '.join,
#     'overall': convert_overall})
# testing_df = test_r.groupby('amazon-id').agg({'helpful': lambda z: np.mean(z), 'reviewText': agg_review, 'summary': agg_review})
#
#
# # Add polarity and subjectivity as additional features
# training_df.insert(
#     1,
#     "pol_rt",
#     training_df[['reviewText']].copy(),
#     True)
# training_df.insert(
#     1,
#     "pol_summary",
#     training_df[['summary']].copy(),
#     True)
# training_df.insert(
#     1,
#     "sub_summary",
#     training_df[['summary']].copy(),
#     True)
#
# testing_df.insert(
#     1,
#     "pol_rt",
#     testing_df[['reviewText']].copy(),
#     True)
# testing_df.insert(
#     1,
#     "pol_summary",
#     testing_df[['summary']].copy(),
#     True)
# testing_df.insert(
#     1,
#     "sub_summary",
#     testing_df[['summary']].copy(),
#     True)
#
# training_df['pol_rt'] = training_df['pol_rt'].apply(lambda x: TextBlob(x).sentiment.polarity)
# training_df['pol_summary'] = training_df['pol_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
# training_df['sub_summary'] = training_df['sub_summary'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
#
# testing_df['pol_rt'] = testing_df['pol_rt'].apply(lambda x: TextBlob(x).sentiment.polarity)
# testing_df['pol_summary'] = testing_df['pol_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
# testing_df['sub_summary'] = testing_df['sub_summary'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
#
#
# # Lemmatizes the words
# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#
#     def __call__(self, articles):
#         return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
#
# # Define necessary variables and help objects for pipeline
# tfidf_transformer = ColumnTransformer(
#     transformers=[
#         (
#             'vect_rt',
#             TfidfVectorizer(
#                 tokenizer=LemmaTokenizer(),
#                 max_df=0.8,
#                 min_df=0,
#                 max_features=20000,
#                 ngram_range=(1, 2)),
#             'reviewText'),
#
#         (
#             'vect_summary',
#             TfidfVectorizer(
#                 tokenizer=LemmaTokenizer(),
#                 max_df=0.85,
#                 min_df=0,
#                 max_features=5000,
#                 ngram_range=(1, 1)),
#             'summary')
#     ],
#     remainder='passthrough'
# )
#
#
# # The following is a test for a LogisticRegression Model
# lr_pipeline = Pipeline([
#     ('tfidf', tfidf_transformer),
#     ('lr', LogisticRegression(max_iter=10000, C=5, solver='liblinear', class_weight={0: 0.6, 1: 0.4})),
# ])
#
#
# # The following performs a 10-Fold CV and calculates the weighted F1 score
# wf1_score = cross_val_score(lr_pipeline, training_df.drop('overall', axis=1), training_df[['overall']].values.ravel(), scoring='f1_weighted', cv=10)
#
# print("Weighted F1: ", wf1_score.mean())
# print("Set: ", wf1_score)
#
#
# # The following performs a 10-Fold CV and calculates the weighted recall and precision scores
# recall_score = cross_val_score(lr_pipeline, training_df.drop('overall', axis=1), training_df[['overall']].values.ravel(), scoring='recall_weighted', cv=10)
# precision_score = cross_val_score(lr_pipeline, training_df.drop('overall', axis=1), training_df[['overall']].values.ravel(), scoring='precision_weighted', cv=10)
#
# print("Recall: ", recall_score.mean())
# print("Precision: ", precision_score.mean())
#
#
# # Sets x_train and y_train to cover all of the training dataset
# x_train, y_train = training_df.drop('overall', axis=1), training_df['overall']
# x_test = testing_df
#
# # Fit model
# lr_pipeline.fit(x_train, y_train)
#
# # Generate and output predictions
# preds = lr_pipeline.predict(x_test)
# output = pd.DataFrame({'amazon-id': x_test.index, 'Awesome': preds})
# output.to_csv('./ExtraCreditPredictions.csv')
#
#
# ############################## Extra Helper Code ##############################
#
# # The following modules are used to determine which classifiers/hyperparameters optimize our results:
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
#
#
# # # Testing additional models for Deliverable #4.
# # models = [
# #     ('AB', AdaBoostClassifier()),
# #     ('XGB', XGBClassifier()),
# #     ('LS', LinearSVC()),
# #     ('LR', LogisticRegression()),
# #     ('RFC', RandomForestClassifier())
# # ]
# #
# # X_train, X_test, y_train, y_test = train_test_split(
# #     training_df.drop('overall', axis=1),
# #     training_df['overall'],
# #     test_size=0.2,
# #     stratify=training_df['overall'])
# #
# # for name, model in models:
# #     clf_pipeline = Pipeline([
# #         ('tfidf', tfidf_transformer),
# #         ('clf', model)
# #     ])
# #
# #     clf_pipeline.fit(X_train, y_train)
# #     pred = clf_pipeline.predict(X_test)
# #     print(name, ": ")
# #     print(classification_report(y_test, pred))
#
#
# # # The following tunes the hyperparameters for TFIDF and Logistic Regression classifier
# # parameters_lr = {
# #     'tfidf__vect_rt__max_df': np.linspace(.8, .95, num=15),
# #     'tfidf__vect_rt__min_df': np.linspace(.01, .2, num=15),
# #     'tfidf__vect_rt__ngram_range': [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)],
# #     'tfidf__vect_rt__max_features': np.linspace(1000, 25000).astype(int),
# #     'tfidf__vect_summary__max_df': np.linspace(0.9, 1, num=20),
# #     'tfidf__vect_summary__min_df': np.linspace(0, .05, num=10),
# #     'tfidf__vect_summary__ngram_range': [(1,1), (1,2)],
# #     'tfidf__vect_summary__max_features': np.linspace(500, 20000).astype(int)
# #     'lr__dual': [True, False],
# #     'lr__penalty': ['l1', 'l2', 'elasticnet', 'none'],
# #     'lr__tol': np.linspace(1e-5, 1e-4, num=2),
# #     'lr__C': np.linspace(1, 20, num=40),
# #     'lr__fit_intercept': [True, False],
# #     'lr__intercept_scaling': [0.001,0.01,0.1,1,10,100],
# #     'lr__class_weight': [{0:.6, 1:.4}, {0:.63, 1:.37}, {0:.67, 1:.33}, {0:.64, 1:.36}, {0:.65, 1:.35}, {0:.66, 1:.34}],
# # }
# #
# # gs_lr = GridSearchCV(
# #     lr_pipeline,
# #     parameters_lr,
# #     scoring='f1_weighted',
# #     cv=5,
# #     verbose=3,
# #     n_jobs=-1
# # )
# #
# # gs_lr.fit(training_df.drop('overall', axis=1), training_df[['overall']].values.ravel())
# #
# # print("The best score: ", gs_lr.best_score_)
# # print("The best hyperparameters: ", gs_lr.best_estimator_.steps)
#
#
# # # Stratified Shuffle Split
# # s = StratifiedShuffleSplit(n_splits=10, random_state=7)
# # X = training_df.drop('overall', axis=1)
# # y = training_df['overall']
# #
# # scores = []
# # for train, test, in s.split(X, y):
# #     X_train, X_test = X.iloc[train], X.iloc[test]
# #     y_train, y_test = y.iloc[train], y.iloc[test]
# #     xgbc_pipeline.fit(X_train, y_train)
# #     pred = xgbc_pipeline.predict(X_test)
# #
# #     score = f1_score(y_test, pred, average='weighted')
# #     scores.append(score)
# #     print(score)
# #
# # print(sum(scores) / len(scores))
#
#
# # # Split training dataset, train and predict the classes, and get the classification report
# # x_train, x_test, y_train, y_test = train_test_split(training_df.drop('overall', axis=1), training_df['overall'], test_size=0.2, random_state=4, stratify=training_df['overall'])
# # lr_pipeline.fit(x_train, y_train)
# # pred = lr_pipeline.predict(x_test)
# # print(classification_report(y_test, pred))
