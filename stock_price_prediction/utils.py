"""
Some magics for stock price prediction notebook.
"""
import numpy as np

def normalize(seq_arr):
    """Normalize sequential datas.

    Formula:
        $$ n_i = (\frac{p_i}{p_0}) - 1 $$
    """
    norm_arr = list()
    for seq in seq_arr:
        norm_seq = [((float(x) / seq[0])- 1) for x in seq]
        norm_arr.append(norm_seq)
    return norm_arr

def data_processing(datas, seq_len=50):
    """Pre-process the datas.

    Inputs:
        datas (np.array): the datas to be processed.
    """
    cuts = round(len(datas) / seq_len)
    processed = list()

    for index in range(cuts):
        processed.append(datas[index: index + (seq_len + 1)])
    
    processed = np.array(normalize(processed))
    train_boundary = round(0.7 * processed.shape[0])
    train = processed[:int(train_boundary), :]
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = processed[int(train_boundary):, :-1]
    y_test = processed[int(train_boundary):, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]