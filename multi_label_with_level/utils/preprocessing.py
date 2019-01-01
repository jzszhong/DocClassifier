import json, string, sys, re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag

from .paths import *
from .loading import *


def preprocess_data(doc_text_list, doc_topic_list, max_ngram=4, outliers=False, scale=False, normalizing=False):
    print('Preprocessing training data...', end='')
    sys.stdout.flush()

    doc_topic_list = MultiLabelBinarizer().fit_transform(doc_topic_list)

    # Handles outliers
    label_index_map = None  # A map points the index of Y to label id
    if not outliers:
        doc_topic_list, label_index_map = remove_outliers(doc_topic_list)

    # Splits training data into training and test sets
    X_train_docs, X_test_docs, Y_train, Y_test = train_test_split(doc_text_list, doc_topic_list, test_size=0.25, random_state=42)
    Y_train, Y_test = stratify_for_rare_label(doc_text_list, doc_topic_list, X_train_docs, X_test_docs, Y_train, Y_test)
    # TF-IDF vectoriser
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, max_ngram), min_df=2, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train_docs)
    X_test = vectorizer.transform(X_test_docs)

    dump_object(vectorizer, "vectorizer")

    # Handles scaling and normalizing
    if scale:
        scaler = StandardScaler(copy=False, with_mean=False)
        X_train = scaler.fit_transform(X_train, Y_train)
        X_test = scaler.transform(X_test)
    if normalizing:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    # Feature selection
    lr_sel_model = OneVsRestClassifier(LogisticRegression())
    lr_sel_model.fit(X_train, Y_train)

    select_model = SelectFromModel(lr_sel_model, prefit=True)
    X_train = select_model.transform(X_train)
    X_test = select_model.transform(X_test)

    dump_object(lr_sel_model, "lr_sel_model")

    print('Split size: ' + format(Y_test.shape[0]/doc_topic_list.shape[0]))
    print('Finished')
    
    return X_train, X_test, Y_train, Y_test, label_index_map

def tokenizer(doc_text):
    def lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return lemmatizer.lemmatize(token, tag)

    # initialize token list
    tokens = []

    lemmatizer = WordNetLemmatizer()
    punctuation = set(string.punctuation)
    stopwords = set(sw.words('english'))
    doc_text = re.sub('[^A-Za-z]', ' ', doc_text)
    
    # split the document into sentences
    for sent in sent_tokenize(doc_text):
        # split the document into tokens and then create part of speech tag for each token
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # preprocess and remove unnecessary characters
            token = token.lower().strip()

            # If stopword, ignore token and continue
            if token in stopwords:
                continue

            # If punctuation, ignore token and continue
            if all(char in punctuation for char in token):
                continue

            # Lemmatize the token and add back to the token
            lemma = lemmatize(token, tag)
            tokens.append(lemma)
    
    return tokens

'''
Prepares training data in a particular format
'''
def prep_training_data(metadata):
    doc_text_list = []
    doc_topic_list = []
    explored_docs = []

    print('Loading training data...', end='')
    sys.stdout.flush()

    for doc_metadata in metadata:
        doc_name = doc_metadata['doc_name']
        if doc_name not in explored_docs:
            doc_topic_list.append(doc_metadata['doc_topic_id'])
            doc_file_path = Path.techone_training_data() + doc_metadata['doc_topic'][0] + '/' + doc_name
            doc_text_list.append(load_doc(doc_file_path))
        explored_docs.append(doc_name)
       
    print('Finished')
    
    return doc_text_list, doc_topic_list

'''
Removes outlier labels whose number of positive samples is less a specific number
'''
def remove_outliers(doc_topic_list, min_samples=100):
    label_index_map = [] # use a array to map the data's label index to its topic id
    outlier_labels = []
    labels_occurence = count_labels_occurence(doc_topic_list)

    for i in range(len(labels_occurence)):
        if labels_occurence[i] < min_samples:
            outlier_labels.append(i)
        else:
            label_index_map.append(i)

    if len(outlier_labels) > 0:
        doc_topic_list = np.delete(doc_topic_list, outlier_labels, axis=1)

    return doc_topic_list, label_index_map

def count_labels_occurence(y):
        num_labels = y.shape[0]
        labels_occurence = (np.arange(num_labels).reshape(1, num_labels) * 0)[0]
        for dataset in y:
            for i in range(y.shape[1]):
                if dataset[i] == 1:
                    labels_occurence[i] += 1
        return labels_occurence


def stratify_for_rare_label(X, y, X_train, X_test, Y_train, Y_test):
    def move_dataset_from_test_to_train(append_index, X, y, X_train, X_test, Y_train, Y_test):
        # Appends the extra datasets to train data by the indices
        X_train.append(X[append_index])
        Y_train = np.append(Y_train, [y[append_index]], axis=0)

        # Gets the index for test data to be removed for Y_test
        rm_index = X_test.index(X[append_index])
        X_test.pop(rm_index)
        Y_test = np.delete(Y_test, rm_index, axis=0)

        return Y_train, Y_test

    min_occurence_prop_ignored = 0.05
    num_dataset = y.shape[0]
    num_train = Y_train.shape[0]
    num_labels = y.shape[1]

    labels_occurence = count_labels_occurence(y)

    # Checks each label
    for i in range(num_labels):
        label_occurence = labels_occurence[i]
        label_occurence_in_train = count_labels_occurence(Y_train)[i]

        # Records the indices of the datasets to be appended to train data
        if label_occurence == 1 and label_occurence_in_train == 0:
            for j in range(num_dataset):
                if y[j, i] == 1 and X[j] not in X_train:
                    Y_train, Y_test = move_dataset_from_test_to_train(j, X, y, X_train, X_test, Y_train, Y_test)
                    num_train = Y_train.shape[0]
                    break

        elif label_occurence == num_dataset - 1 and label_occurence_in_train == num_train:
            for j in range(num_dataset):
                if y[j, i] == 0 and X[j] not in X_train:
                    Y_train, Y_test = move_dataset_from_test_to_train(j, X, y, X_train, X_test, Y_train, Y_test)
                    num_train = Y_train.shape[0]
                    break

        elif label_occurence / num_dataset <= min_occurence_prop_ignored:
            # If it occurs more than once but smaller the occurence that can be ignored,
            # only appends half of them to train
            limit = round(label_occurence / 2)
            if label_occurence_in_train < limit:
                for j in range(num_dataset):
                    if label_occurence_in_train < limit:
                        if y[j, i] == 1:
                            if X[j] not in X_train:
                                Y_train, Y_test = move_dataset_from_test_to_train(j, X, y, X_train, X_test, Y_train,
                                                                                  Y_test)
                                label_occurence_in_train += 1
                                num_train += 1
                    else:
                        break

        elif (
                num_dataset - label_occurence) / num_dataset <= min_occurence_prop_ignored and num_dataset - label_occurence > 0:
            # If it occurs more than once but smaller the occurence that can be ignored,
            # only appends half of them to train
            limit = round(label_occurence / 2)
            if num_train - label_occurence_in_train < limit:
                for j in range(num_dataset):
                    if num_train - label_occurence_in_train < limit:
                        if y[j, i] == 0:
                            if X[j] not in X_train:
                                Y_train, Y_test = move_dataset_from_test_to_train(j, X, y, X_train, X_test, Y_train,
                                                                                  Y_test)
                                num_train += 1
                    else:
                        break

    return Y_train, Y_test
