import time

from utils.preprocessing import *
from utils.predicting import *


def build_models_with_level(build_model, clf_name):
    t0 = time.time()

    accuracies = []
    Y_pred_full = []

    # Prepares full data
    metadata = load_metadata(Path.techone_training_data() + 'metadata.json')
    doc_text_list, doc_topic_list = prep_training_data(metadata)
    X_train, X_test, Y_train, Y_test, label_index_map = preprocess_data(doc_text_list, doc_topic_list)

    # Trains models label by label
    for i in range(Y_test.shape[1]):
        model, accuracy, Y_pred = build_model(X_train, X_test, Y_train[:, i], Y_test[:, i])

        Y_pred_full.append(Y_pred)

        accuracies.append(accuracy)

        label_id = label_index_map[i]
        dump_object(model, '/model' + str(label_id), clf_name)

    # Calculates average label accuracy
    sum = 0
    for a in accuracies:
        sum += a
    avg_accuracy = sum / len(accuracies)
    print("\nAverage accuracy:", avg_accuracy)

    # Gets overall test accuracy
    Y_pred_full = np.array(Y_pred_full).transpose()
    strict_accuracy = get_accuracy(Y_pred_full, Y_test)
    print(strict_accuracy)

    t2 = time.time()
    print ('Overall Took {0} minutes'.format((t2-t0) / 60.0))


