# Requires Python 3.5 or above
import numpy as np
import re
import io
import os
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


NUM_FOLDS = 5               # Value of K in K-fold Cross Validation
NUM_CLASSES = 4             # Number of classes - Angry, Haha, Sad, Love
MAX_NB_WORDS = 20000        # Limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 100   # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = 300         # The dimension of the word embeddings
BATCH_SIZE = 200            # The batch size to be chosen for training the model.
LSTM_DIM = 128              # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.2               # Fraction of the units to drop for the linear transformation of the inputs.
LEARNING_RATE = 0.003
NUM_EPOCHS = 10             # Number of epochs to train a model for

# Path to training and testing data file.
trainDataPath = "datafiles\\train_balanced_normalized_dataset.txt"
testDataPath = "datafiles\\test_balanced_normalized_dataset.txt"

# Output file that will be generated.
solutionPath = "datafiles\\test_balanced_normalized_dataset_predictions.txt"

# Path to directory where GloVe file is saved.
gloveFile = "glove\\es\\glove-sbwc.i25.txt"

script_directory = os.path.dirname(__file__)

label2emotion = {0: "ANGRY", 1: "SAD", 2: "HAHA", 3: "LOVE"}
emotion2label = {"ANGRY": 0, "SAD": 1, "HAHA": 2, "LOVE": 3}


def preprocess_data(data_file_path, mode, context=True):
    """
    Load data from a file, preprocess it, and extract relevant components.

    Args:
        data_file_path (str): The path to the data file to be processed.
        mode (str): Specifies the mode of operation. In "train" mode, labels are returned; in "test" mode,
        labels are not returned.
        context (bool): Indicates whether context information is included in the output.

    Returns:
        indices (list): A list of unique conversation ID's.
        conversations (list): A list of processed conversations, where each turn is separated by the "<eos>" tag.
        labels (list, optional): Only available in "train" mode. A list of labels associated with the conversations.
    """
    indices = []
    conversations = []
    labels = []
    with io.open(data_file_path, encoding="utf8") as f_input:
        f_input.readline()
        for line in f_input:
            # Convert multiple instances of . ? ! , to single instance and adds whitespace around such punctuation
            repeated_chars = ['.', '?', '!', ',']
            for c in repeated_chars:
                line_split = line.split(c)
                while True:
                    try:
                        line_split.remove('')
                    except:
                        break
                c_space = ' ' + c + ' '
                line = c_space.join(line_split)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 documents and label
                label = emotion2label[line[4]]
                labels.append(label)

            if context:
                conv = ' <eos> '.join(line[1:4])
            else:
                conv = ' <eos> '.join([line[3]])

            # Remove any duplicate spaces
            duplicate_space_pattern = re.compile(r'\ +')
            conv = re.sub(duplicate_space_pattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def get_metrics(predictions, ground):
    """
    Calculate and display various metrics based on predicted labels and corresponding ground truth labels.

    Args:
        predictions (numpy.ndarray): Model output containing predicted probabilities for each class.
            Shape should be [# of samples, NUM_CLASSES].
        ground (numpy.ndarray): Ground truth labels, converted to one-hot encodings. Each sample's label is represented
            as a one-hot vector. For example, a sample belonging to the "Happy" class will be [0, 1, 0, 0].

    Returns:
        accuracy (float): Average accuracy of the predictions.
        microPrecision (float): Precision calculated on a micro level.
        microRecall (float): Recall calculated on a micro level.
        microF1 (float): Harmonic mean of microPrecision and microRecall. A higher value indicates better classification.
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discrete_predictions = to_categorical(predictions.argmax(axis=1), num_classes=predictions.shape[1])

    true_positives = np.sum(discrete_predictions * ground, axis=0)
    false_positives = np.sum(np.clip(discrete_predictions - ground, 0, 1), axis=0)
    false_negatives = np.sum(np.clip(ground - discrete_predictions, 0, 1), axis=0)

    print("True Positives per class : ", true_positives)
    print("False Positives per class : ", false_positives)
    print("False Negatives per class : ", false_negatives)

    # ------------- Macro level calculation ---------------
    macro_precision = 0
    macro_recall = 0
    for c in range(0, NUM_CLASSES):
        precision = true_positives[c] / (true_positives[c] + false_positives[c])
        macro_precision += precision
        recall = true_positives[c] / (true_positives[c] + false_negatives[c])
        macro_recall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macro_precision /= 4
    macro_recall /= 4
    macro_f1 = (2 * macro_recall * macro_precision) / (macro_precision + macro_recall) \
        if (macro_precision + macro_recall) > 0 else 0
    print("Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macro_precision, macro_recall, macro_f1))

    # ------------- Micro level calculation ---------------
    true_positives = true_positives.sum()
    false_positives = false_positives.sum()
    false_negatives = false_negatives.sum()

    print("Micro TP : %d, FP : %d, FN : %d" % (true_positives, false_positives, false_negatives))

    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)

    micro_f1 = (2 * micro_recall * micro_precision) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : "
          "%.4f, Micro F1 : %.4f" % (accuracy, micro_precision, micro_recall, micro_f1))
    return accuracy, micro_precision, micro_recall, micro_f1


def get_embedding_matrix(word_index):
    """
    Create an embedding matrix using a word-index mapping. Each row in the matrix corresponds to a word,
    and the matrix is populated with 300-dimensional GloVe embeddings.

    Args:
        word_index (dict): A dictionary containing (word : index) pairs, typically generated using a tokenizer.

    Returns:
        embedding_matrix (numpy.ndarray): A matrix where each row represents a word's GloVe embedding,
            and the matrix has a shape of [vocab_size, embedding_dim].
    """
    embeddings_index = {}
    # Load the embedding vectors from their GloVe file
    with io.open(os.path.join(script_directory, gloveFile), encoding="utf8", errors='ignore') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding_vector

    print('Found %s word vectors.' % len(embeddings_index))

    # Minimum word index of any word is 1.
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_model(embedding_matrix):
    """
    Build the architecture of a LSTM-based model.

    Args:
        embedding_matrix (numpy.ndarray): The embedding matrix to be loaded into the embedding layer.

    Returns:
        model: A model with an LSTM-based architecture.
    """
    embedding_layer = Embedding(embedding_matrix.shape[0], EMBEDDING_DIM,
                                weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    rms_prop = optimizers.RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms_prop,
                  metrics=['acc'])
    return model


def main():

    print("Processing training data...")
    train_indices, train_texts, labels = preprocess_data(
        os.path.join(script_directory, trainDataPath), mode="train", context=False)

    print("Processing test data...")
    test_indices, test_texts = preprocess_data(os.path.join(script_directory, testDataPath), mode="test", context=False)
    print(test_indices)
    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    print("Creating embedding matrix...")
    embedding_matrix = get_embedding_matrix(word_index)

    data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(train_indices)
    data = data[train_indices]
    labels = labels[train_indices]

    # Perform k-fold cross validation
    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}

    print("Starting k-fold cross validation...")
    for k in range(NUM_FOLDS):
        print('-' * 50)
        print("Fold %d/%d" % (k + 1, NUM_FOLDS))
        validation_size = int(len(data) / NUM_FOLDS)
        index1 = validation_size * k
        index2 = validation_size * (k + 1)

        x_train = np.vstack((data[:index1], data[index2:]))
        y_train = np.vstack((labels[:index1], labels[index2:]))
        x_val = data[index1:index2]
        y_val = labels[index1:index2]
        print("Building model...")
        model = build_model(embedding_matrix)
        model.fit(x_train, y_train,
                  validation_data=(x_val, y_val),
                  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        predictions = model.predict(x_val, batch_size=BATCH_SIZE)
        accuracy, micro_precision, micro_recall, micro_f1 = get_metrics(predictions, y_val)
        metrics["accuracy"].append(accuracy)
        metrics["microPrecision"].append(micro_precision)
        metrics["microRecall"].append(micro_recall)
        metrics["microF1"].append(micro_f1)

    print("\n============= Metrics =================")
    print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
    print("Average Cross-Validation Micro Precision : %.4f" % (
                sum(metrics["microPrecision"]) / len(metrics["microPrecision"])))
    print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"]) / len(metrics["microRecall"])))
    print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"]) / len(metrics["microF1"])))

    print("\n======================================")

    print("Retraining model on entire data to create solution file")
    model = build_model(embedding_matrix)
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5' % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(test_data, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(os.path.join(script_directory, solutionPath), "w", encoding="utf8") as f_out:
        f_out.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(os.path.join(script_directory, testDataPath), encoding="utf8") as f_in:
            f_in.readline()
            for lineNum, line in enumerate(f_in):
                f_out.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                f_out.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


if __name__ == '__main__':
    main()
