import sys, os
import numpy as np
from utils.paths import *
from utils.loading import *
from utils.preprocessing import *
from sklearn.feature_selection import SelectFromModel


def predict(model_name, X):
    def load_models(model_name, label_ids):
        models = {}
        model_dir = 'models/' + model_name

        for label_id in label_ids:
            model_path = model_dir + '/model' + str(label_id)

            if os.path.exists(model_path):
                models[label_id] = load_object(model_path)

        return models

    def get_probas(models, X):
        probas = {}
        for label_id in models.keys():
            model = models[label_id]
            probas[label_id] = round(model.predict_proba(X)[0][1], 3)
        return probas

    prediction = []
    superlabels, child_label_tree = create_supertopic_list_and_childtopic_tree()

    # Gets probability for each super label and shows them
    supermodels = load_models(model_name, superlabels)
    superlabel_probas = get_probas(supermodels, X)

    print('Probabilities for the doc to be classified with the following tags:')
    for label_id in superlabel_probas.keys():
        print(label_id, ':', superlabel_probas[label_id])

    superlabel_id = input("Choose one that fits the doc: ")
    prediction.append(superlabel_id)

    # Check if there is child label for that super label, if so, predicts it too
    if superlabel_id in child_label_tree.keys():
        childlabels = child_label_tree[superlabel_id]
        childmodels = load_models(model_name, childlabels)
        childlabel_probas = get_probas(childmodels, X)

        print('Probabilities for the doc to be classified with the following child tags:')
        for label_id in childlabel_probas.keys():
            print(label_id, ':', childlabel_probas[label_id])

        childlabel_id = input("Choose one that fits the doc: ")
        prediction.append(childlabel_id)

    return prediction

def predict_silence(model_name, X):
    def load_models(model_name, label_ids):
        models = {}
        model_dir = 'models/' + model_name

        for label_id in label_ids:
            model_path = model_dir + '/model' + str(label_id)

            if os.path.exists(model_path):
                models[label_id] = load_object(model_path)

        return models

    def get_probas(models, X):
        probas = {}
        for label_id in models.keys():
            model = models[label_id]
            probas[label_id] = round(model.predict_proba(X)[0][1], 3)
        return probas

    predictions = []
    superlabels, child_label_tree = create_supertopic_list_and_childtopic_tree()

    # Gets probability for each super label and shows them
    supermodels = load_models(model_name, superlabels)
    superlabel_probas = get_probas(supermodels, X)

    superlabel_probas_keys = list(superlabel_probas.keys())
    top5_superlabel_idx = np.array(list(superlabel_probas.values())).argsort()[::-1][:5]

    for label_idx in top5_superlabel_idx:
        label_id = superlabel_probas_keys[label_idx]
        predictions.append({
            'id': label_id,
            'proba': superlabel_probas[label_id]
        })

    for label_idx in top5_superlabel_idx:
        superlabel_id = superlabel_probas_keys[label_idx]

        # Check if there is child label for that super label, if so, predicts it too
        if superlabel_id in child_label_tree.keys():
            childlabels = child_label_tree[superlabel_id]
            childmodels = load_models(model_name, childlabels)
            childlabel_probas = get_probas(childmodels, X)

            childlabel_probas_keys = list(childlabel_probas.keys())
            top5_childlabel_idx = np.array(list(childlabel_probas.values())).argsort()[::-1][:5]

            for label_idx in top5_childlabel_idx:
                label_id = childlabel_probas_keys[label_idx]
                predictions.append({
                    'id': label_id,
                    'proba': childlabel_probas[label_id]
                })

    return predictions

def get_tag_silence(predictions):
    results = []
    topic_list = load_metadata(Path.techone_training_data()+'topic_dict.json')
    for prediction in predictions:
        results.append({
            'tag': topic_list[str(prediction['id'])],
            'proba': prediction['proba']
        })

    return results

def get_tag(prediction):
    results = []
    topic_list = load_metadata(Path.techone_training_data()+'topic_dict.json')

    for label_id in prediction:
        results.append(topic_list[str(label_id)])

    return results

def LR(text):
    lr_sel_model = load_object('models/lr_sel_model')
    vectorizer = load_object('models/vectorizer')
    select_model = SelectFromModel(lr_sel_model, prefit=True)

    data = vectorizer.transform([text])
    data = select_model.transform(data)

    return data


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        text = file.read()
        data = LR(text)
        print(get_tag(predict('LR', data)))
