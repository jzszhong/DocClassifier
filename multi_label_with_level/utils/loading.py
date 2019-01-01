import json, os
import numpy as np
from sklearn.externals import joblib

from .paths import *


def load_metadata(metadata_path):
    with open(metadata_path) as raw_data:
        metadata = json.load(raw_data)

    return metadata

def load_doc(doc_path):
    with open(doc_path, 'r') as file:
        return file.read()

def dump_object(object, name, path=None):
    file_dir = 'models/'
    if path is not None:
        file_dir += path

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    object_file_path = file_dir + name

    joblib.dump(object, object_file_path, compress=True, protocol=4)

def load_object(path):
    return joblib.load(path)

'''
Creates a list with the id of level-1 label and a tree map showing the child labels from particular level-1 labels
'''
def create_supertopic_list_and_childtopic_tree():
    def get_topic_id_by_name(topic_list, topic_name):
        for topic_id in topic_list.keys():
            if topic_list[topic_id] == topic_name:
                return topic_id

    supertopics = []
    childtopic_tree = {}

    topic_list = load_metadata(Path.techone_training_data() + 'topic_dict.json')
    topic_names = topic_list.values()

    for topic_id in topic_list.keys():
        topic_name = topic_list[topic_id]
        topic_parts = topic_name.split('/')

        # For super-topics
        if len(topic_parts) == 1:
            supertopics.append(int(topic_id))

        # For child-topics
        else:
            supertopic = topic_parts[0]

            # If a topic's supertopic has no samples, treat it as a super-topic
            if supertopic not in topic_names:
                supertopics.append(int(topic_id))

            # Otherwise treats it as a complex topic with a supertopic and childtopic
            else:
                supertopic_id = get_topic_id_by_name(topic_list, supertopic)

                if int(supertopic_id) not in childtopic_tree.keys():
                    childtopic_tree[int(supertopic_id)] = [int(topic_id)]
                else:
                    childtopic_tree[int(supertopic_id)].append(int(topic_id))

    return supertopics, childtopic_tree

def get_partial_Y(Y_train, Y_test, partial_labels):
    print(Y_train.shape[1])
    num_train_samples = Y_train.shape[0]
    num_test_samples = Y_test.shape[0]
    num_labels = len(partial_labels)

    Y_train_sup = np.zeros([num_train_samples, num_labels])
    Y_test_sup = np.zeros([num_test_samples, num_labels])

    for j in range(num_labels):
        label_id = partial_labels[j] - 1
        for i in range(num_train_samples):
            if Y_train[i, label_id] == 1:
                Y_train_sup[i, j] = 1
        for i in range(num_test_samples):
            if Y_test[i, label_id] == 1:
                Y_test_sup[i, j] = 1

    return Y_train_sup, Y_test_sup
