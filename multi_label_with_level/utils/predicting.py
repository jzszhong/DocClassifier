

def get_accuracy(Y_pred, Y):
    num_datasets = Y.shape[0]
    num_labels = Y.shape[1]
    num_correct = 0
    for i in range(num_datasets):
        all_labels_correct = True
        for j in range(num_labels):
            if not Y_pred[i, j] == Y[i, j]:
                all_labels_correct = False
                break
        if all_labels_correct:
            num_correct += 1
            
    return num_correct/num_datasets

def unlabelled_rate(Y_pred):
    num_pred = Y_pred.shape[0]
    num_unlabelled = 0
    for result in Y_pred:
        if 1 not in result:
            num_unlabelled += 1
    return num_unlabelled / num_pred
