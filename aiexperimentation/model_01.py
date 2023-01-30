import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded_sequence = sequence + [0] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        padded_sequences.append(padded_sequence)
    return padded_sequences

# Preprocess the dataset
conversations = [{'messages': [{'sender': 'user',
                                'text': 'Hello, how are you doing?'},
                               {'sender': 'bot',
                                'text': "I'm doing well, thanks for asking. How about you?"},
                               {'sender': 'user',
                                'text': 'I am doing okay. Can you tell me more about your services?'},
                               {'sender': 'bot',
                                'text': 'Certainly! We offer a wide range of services, including website design, social media management, and content creation. Is there anything specific you are interested in?'},
                               {'sender': 'user',
                                'text': 'Yes, I am interested in getting a new website for my business. Can you give me a quote?'},
                               {'sender': 'bot',
                                'text': 'Sure, I would be happy to provide you with a quote. Could you tell me more about your business and what you are looking for in a website?'}]},
                 {'messages': [{'sender': 'user',
                                'text': 'Hi, I am looking for a new job. Do you have any openings?'},
                               {'sender': 'bot',
                                'text': 'Hello! We do not currently have any openings, but we would be happy to keep your resume on file for future opportunities. Could you tell me more about your experience and what you are looking for in a job?'},
                               {'sender': 'user',
                                'text': 'I have 5 years of experience in marketing and I am looking for a full-time position with a company that values creativity and innovation. Do you have any positions that might fit my criteria?'},
                               {'sender': 'bot',
                                'text': 'Thank you for sharing your experience and interests with us. We will keep your resume on file and notify you if any relevant positions become available. In the meantime, you may want to check out our job board or follow us on social media to stay updated on our latest job openings.'}]}]

# Tokenize the text and create vocabulary
all_text = []
for conversation in conversations:
    for message in conversation['messages']:
        all_text.append(message['text'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
vocab_size = len(tokenizer.word_index) + 1

# Encode and pad the sequences
encoded_text = tokenizer.texts_to_sequences(all_text)
max_length = max([len(sequence) for sequence in encoded_text])
padded_text = pad_sequences(encoded_text, max_length)

# Split the dataset into training and testing sets
split_index = int(len(padded_text) * 0.8)
X_train, X_test = padded_text[:split_index], padded_text[split_index:]

y_train, y_test = padded_text[:split_index], padded_text[split_index:]
# Create training and testing datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(vocab_size, activation='softmax'))

# Compile and fit the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

# Evaluate the model
scores = model.evaluate(test_dataset)
print("Accuracy:", scores[1])
