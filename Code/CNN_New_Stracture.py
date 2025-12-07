#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import numpy as np
import copy
import os
import math
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D


# # Initialize

# In[3]:


def import_uwb_dataset_all_part():

    filename = 'combined_file.csv'
    filename2 = 'combined_file2.csv'
    filename3 = 'combined_file3.csv'
    filename4 = 'combined_file4.csv'
    try:
        df = pd.read_csv(filename, sep=',', header=0)
        df2 = pd.read_csv(filename2, sep=',', header=0)
        df3 = pd.read_csv(filename3, sep=',', header=0)
        df4 = pd.read_csv(filename4, sep=',', header=0)
        data1 = df.values
        data2 = df2.values
        data3 = df3.values
        data4 = df4.values
        return data1,data2,data3,data4
    except FileNotFoundError:
        print("The file uwb_dataset_all_part.csv was not found in the current directory.")
        return None

data1,data2,data3,data4 = import_uwb_dataset_all_part()
data = np.vstack((data1,data2,data3,data4))
np.random.shuffle(data)
data_size = data.shape[0]
train_split = int(0.8 * data_size)  # 80% for training
test_split = data_size - train_split  # Remaining 20% for testing

# Split the data
train_data = data[:train_split]
test_data = data[train_split:]

# Divide Data
X_train = train_data[:, 1:].T
Y_train = train_data[:, 0]
Y_train = Y_train.reshape(1, train_split)

X_test = test_data[:, 1:].T
Y_test = test_data[:, 0]
Y_test = Y_test.reshape(1, test_split)
# shape
print ("Shape of : X_train = " + str(X_train.shape))
print ("Shape of : Y_train = " + str(Y_train.shape))
print ("Shape of : X_test = " + str(X_test.shape))
print ("Shape of : Y_test = " + str(Y_test.shape))
mean = np.mean(X_test,axis=0)
print(mean)
# Examples of NLOS LoS
Index_NLOS = 248
Index_LOS = 139
fig, ax = plt.subplots()
ax.plot(X_train[:,Index_NLOS]/2,label='1', color='red')
plt.plot(X_train[:,Index_LOS], label='2', color='blue')
plt.xlim(600, 1000)

ax.set_xlabel("Time (samples)")
ax.set_ylabel("Received signal")
ax.set_title("CIR")
plt.show()
# number of data
m_train =X_train.shape[1]
m_test =X_test.shape[1]
mean = np.mean(X_test,axis=0)
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of training examples: m_test = " + str(m_test))
print (mean)
# Standard
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
mean = np.mean(X_test,axis=0)
print(mean)
def load_training_data():
    # Load your training data (X_train, Y_train)
    return X_train, Y_train

def load_testing_data():
    # Load your testing data (X_test, Y_test)
    return X_test, Y_test


# # FC1=128,FC2=64 DO=0.5
# 
# 
# 

# In[ ]:


# Assuming the functions load_training_data and load_testing_data are defined elsewhere to load your data
current_directory = os.getcwd()
TRAIN_DIR = os.path.join(current_directory, 'my_checkpoint_directory')

BATCH_SIZE = 200
MAX_STEPS = 20000
start_time = time.time()
def build_model(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size, 1), input_shape=(input_size,)),
        tf.keras.layers.Conv1D(10, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(20, 5, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Conv1D(20, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(40, 4, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model

def main():
    X_train, Y_train = load_training_data()  # Replace with your data loading logic
    X_test, Y_test = load_testing_data()      # Replace with your data loading logic

    # Transpose the data arrays to match the new shape
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    input_data_size = X_train.shape[1]  # Assuming X_train is a NumPy array

    try:
        os.makedirs(TRAIN_DIR)
    except OSError:
        if os.path.exists(TRAIN_DIR):
            # We are nearly safe
            pass
        else:
            # There was an error on creation
            raise

    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')

    model = build_model(input_data_size)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Convert labels to categorical format
    Y_train_categorical = tf.keras.utils.to_categorical(Y_train, num_classes=2)
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, num_classes=2)

    # Train the model
    model.fit(X_train, Y_train_categorical, epochs=40, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test_categorical))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test_categorical)

    # Save the model
    tf.keras.models.save_model(model, os.path.join(TRAIN_DIR, 'my_model'), save_format='tf')


if __name__ == "__main__":
    main()
end_time = time.time()
time_spent = end_time - start_time
print("Time spent:", time_spent)


# # FC1=128,FC2=128 DO=0.8

# In[ ]:


import os
import time
import math
import tensorflow as tf
import numpy as np

# Assuming the functions load_training_data and load_testing_data are defined elsewhere to load your data
start_time = time.time()
current_directory = os.getcwd()
TRAIN_DIR = os.path.join(current_directory, 'my_checkpoint_directory')

BATCH_SIZE = 200
MAX_STEPS = 20000

def build_model(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size, 1), input_shape=(input_size,)),
        tf.keras.layers.Conv1D(10, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(20, 5, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Conv1D(20, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(40, 4, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model

def main():
    X_train, Y_train = load_training_data()  # Replace with your data loading logic
    X_test, Y_test = load_testing_data()      # Replace with your data loading logic

    # Transpose the data arrays to match the new shape
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    input_data_size = X_train.shape[1]  # Assuming X_train is a NumPy array

    try:
        os.makedirs(TRAIN_DIR)
    except OSError:
        if os.path.exists(TRAIN_DIR):
            # We are nearly safe
            pass
        else:
            # There was an error on creation
            raise

    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')

    model = build_model(input_data_size)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Convert labels to categorical format
    Y_train_categorical = tf.keras.utils.to_categorical(Y_train, num_classes=2)
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, num_classes=2)

    # Train the model
    model.fit(X_train, Y_train_categorical, epochs=40, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test_categorical))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test_categorical)

    # Save the model
    tf.keras.models.save_model(model, os.path.join(TRAIN_DIR, 'my_model'), save_format='tf')


if __name__ == "__main__":
    main()
end_time = time.time()
time_spent = end_time - start_time
print("Time spent:", time_spent)


# # Evaluate the model

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

X_testt = np.transpose(X_test)
Y_testt = np.transpose(Y_test)
X_traint = np.transpose(X_train)
Y_traint = np.transpose(Y_train)

# Assuming you have already trained and evaluated your Keras model and stored it in 'model'
Y_test_categorical = tf.keras.utils.to_categorical(Y_testt, num_classes=2)
Y_train_categorical = tf.keras.utils.to_categorical(Y_testt, num_classes=2)
# Load the saved model
model = tf.keras.models.load_model(os.path.join(TRAIN_DIR, 'my_model'))

# Evaluate the model
loss, accuracy = model.evaluate(X_testt, Y_test_categorical)

# Make predictions on the test data using your trained model
predictions = model.predict(X_testt)

# Convert categorical labels back to original form if needed
# For example, if Y_test_categorical is one-hot encoded, you might need to convert it back
test_labels = np.argmax(Y_test_categorical, axis=1)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)

# Calculate precision and recall for each class (assuming binary classification)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# In[ ]:


import tensorflow as tf
import os
import numpy as np

# Assuming TRAIN_DIR and BATCH_SIZE are defined as in the previous code snippet

def load_model_and_data():
    # Load the saved model
    model = tf.keras.models.load_model(os.path.join(TRAIN_DIR, 'my_model'))

    # Assuming you have a function to load your data
    X_new_data, Y_new_data = load_new_data()  # Replace with your new data loading logic

    # Transpose the new data arrays to match the model input shape
    X_new_data = np.transpose(X_new_data)
    Y_new_data = np.transpose(Y_new_data)

    # Convert labels to categorical format
    Y_new_categorical = tf.keras.utils.to_categorical(Y_new_data, num_classes=2)

    return model, X_new_data, Y_new_categorical

def continue_training(model, X_new_data, Y_new_categorical):
    # Compile the model if needed
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Continue training with new data
    model.fit(X_new_data, Y_new_categorical, epochs=10, batch_size=BATCH_SIZE, validation_split=0.2)

    return model

def predict_with_model(model, X_data):
    # Use the model for prediction
    predictions = model.predict(X_data)

    return predictions

if __name__ == "__main__":
    # Load the model and new data
    loaded_model, new_X_data, new_Y_data = load_model_and_data()

    # Continue training if desired
    trained_model = continue_training(loaded_model, new_X_data, new_Y_data)

    # Example: Using the trained model for prediction
    predictions = predict_with_model(trained_model, new_X_data)
    print(predictions)


# # FC1=128,FC2=64 DO=0.5 model saved

# In[ ]:


# Assuming the functions load_training_data and load_testing_data are defined elsewhere to load your data
current_directory = os.getcwd()
TRAIN_DIR = os.path.join(current_directory, 'my_checkpoint_directory2')

BATCH_SIZE = 200
MAX_STEPS = 20000
start_time = time.time()
def build_model(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size, 1), input_shape=(input_size,)),
        tf.keras.layers.Conv1D(10, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(20, 5, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Conv1D(20, 4, activation='relu', padding='valid'),
        tf.keras.layers.Conv1D(40, 4, activation='relu', padding='same', strides=2),
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model

def main():
    X_train, Y_train = load_training_data()  # Replace with your data loading logic
    X_test, Y_test = load_testing_data()      # Replace with your data loading logic


    # Transpose the data arrays to match the new shape
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    input_data_size = X_train.shape[1]  # Assuming X_train is a NumPy array

    try:
        os.makedirs(TRAIN_DIR)
    except OSError:
        if os.path.exists(TRAIN_DIR):
            # We are nearly safe
            pass
        else:
            # There was an error on creation
            raise

    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint')
    input_data_size = X_train.shape[1]
    model = build_model(input_data_size)
    model_json = model.to_json()
    with open(os.path.join(TRAIN_DIR, "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Convert labels to categorical format
    Y_train_categorical = tf.keras.utils.to_categorical(Y_train, num_classes=2)
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, num_classes=2)

    # Train the model
    model.fit(X_train, Y_train_categorical, epochs=40, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test_categorical))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test_categorical)
    model.save_weights(os.path.join(TRAIN_DIR, 'my_model_weights'))
    model.save(os.path.join(TRAIN_DIR, 'complete_model'), save_format='tf')


    # Save the model
    tf.keras.models.save_model(model, os.path.join(TRAIN_DIR, 'my_model'), save_format='tf')


if __name__ == "__main__":
    main()
end_time = time.time()
time_spent = end_time - start_time
print("Time spent:", time_spent)


# # Predict

# In[ ]:


# Load your X_TRAIN2 data (replace this with your data loading logic)
# Assuming X_TRAIN2 has the same shape as X_train used during training
# X_TRAIN2 = ...

# Load the entire saved model (architecture, weights, optimizer state)
TRAIN_DIR = 'my_checkpoint_directory'  # Modify this to your saved directory
loaded_model_path = os.path.join(TRAIN_DIR, 'complete_model')

loaded_model = tf.keras.models.load_model(loaded_model_path)

# Preprocess X_TRAIN2 as needed (assuming it has the same preprocessing as X_train)
# X_TRAIN2_processed = ...

# Make predictions on X_TRAIN2
predictions = loaded_model.predict(X_TRAIN2_processed)

# Your predictions are now stored in the 'predictions' variable

