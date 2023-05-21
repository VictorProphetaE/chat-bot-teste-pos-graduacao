import nltk
import tflearn
import tensorflow as tf
import random
import json
import pickle
import glob

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import numpy as np

stemmer = LancasterStemmer()


def merge_json_files():
    finaljson = {"intents": []}
    unique_patterns = set()

    for f in glob.glob("*.json"):
        with open(f, "r") as file:
            data = json.load(file)
            for intent in data["intents"]:
                patterns = tuple(intent["patterns"])
                if patterns not in unique_patterns:
                    unique_patterns.add(patterns)
                    finaljson["intents"].append(intent)

    with open("merged.json", "w") as outfile:
        json.dump(finaljson, outfile, indent=2)


def preprocess_data(data):
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    return words, labels, np.array(training), np.array(output)


def train_model(training, output, num_epochs=600):
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=num_epochs, batch_size=8, show_metric=True)
        model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def add_new_word():
    try:
        with open("storedata.json") as f:
            data = json.load(f)
        while True:
            word = input("Entre com uma nova palavra: (ou digite 'sair' para sair) ")
            if word.lower() == "exit":
                break
            else:
                word2 = input("Entre com uma resposta para a nova palavra: ")
                extend1 = True
                extend2 = True
                for intent in data["intents"]:
                    if extend1:
                        intent["patterns"].append(word)
                        extend1 = False
                for intent in data["intents"]:
                    if extend2:
                        intent["responses"].append(word2)
                        extend2 = False

        with open("storedata.json", "w+") as outfile:
            json.dump(data, outfile, indent=2)
    except:
        storedata = {"intents": []}

        while True:
            word = input("Entre com uma nova palavra: (ou digite 'sair' para sair) ")
            if word.lower() == "exit":
                break
            else:
                word2 = input("Entre com uma resposta para a nova palavra: ")
                extend1 = True
                extend2 = True
                if bool(storedata["intents"]):
                    for intent in storedata["intents"]:
                        if extend1:
                            intent["patterns"].append(word)
                            extend1 = False
                    for intent in storedata["intents"]:
                        if extend2:
                            intent["responses"].append(word2)
                            extend2 = False
                else:
                    storedata["intents"].append(
                        {"tag": "New", "patterns": [word], "responses": [word2], "context_set": ""}
                    )

        with open("storedata.json", "w") as outfile:
            json.dump(storedata, outfile, indent=2)


def chat():
    print("Fale com o bot (digite 'sair' para sair)!")
    while True:
        user_input = input("Eu: ")
        if user_input.lower() == "quit":
            break

        results = model.predict([bag_of_words(user_input, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for intent in data["intents"]:
            if intent["tag"] == tag:
                print("Bot:", random.choice(intent["responses"]))
			
# Merge JSON files
merge_json_files()

# Load and preprocess data
with open("merged.json") as f:
    data = json.load(f)

words, labels, training, output = preprocess_data(data)

# Train the model
train_model(training, output)

# Load the trained model
model = tflearn.DNN(net)
model.load("model.tflearn")

# Start the chat
chat()