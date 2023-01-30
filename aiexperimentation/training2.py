import json
import pickle
import numpy as np
import nltk
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()

# Load the JSON file
intents = json.loads(open("intent.json").read())

# Create empty lists to store data
words = []
classes = []
documents = []
ignore_words = ['?']

# Loop through the intents in the JSON file
for intent in intents['intents']:
    # Loop through the text patterns for each intent
    for pattern in intent['text']:
        # Tokenize the pattern
        word_list = nltk.word_tokenize(pattern)
        # Add the words to the list of words
        words.extend(word_list)
        # Create a document for the pattern
        documents.append((word_list, intent['intent']))
        # Add the intent to the list of classes
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

# Lemmatize the words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Remove duplicates
words = sorted(list(set(words)))

# Save the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Find the maximum length of the elements in training
max_length = max([len(x) for x in training])

# Loop through the elements in training and pad them with zeros
# so that they all have the same length as the longest element
for element in training:
    while len(element) < max_length:
        element.append(0)

# Convert the training list into a NumPy array
training = np.array(training)

# Flatten the elements in the training list
flat_training = [item for sublist in training for item in sublist]

# Convert the flat_training list into a NumPy array
flat_training = np.array(flat_training)

# Split the data into the input and output sets
train_x = flat_training[:, 0]
train_y = flat_training[:, 1]


# Create a Sequential machine learning model
model = Sequential()

# Add layers to the model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model to the training data
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("chatbotmodel.h5", hist)
