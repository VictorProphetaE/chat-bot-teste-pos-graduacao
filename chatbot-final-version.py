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
        bag = [1 if stemmer.stem(w) in doc else 0 for w in words]
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
    bag = [1 if stemmer.stem(word.lower()) in word_tokenize(s) else 0 for word in words]
    return np.array(bag)

def add_new_word():
    try:
        with open("storedata.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"intents": []}

    while True:
        word = input("Entre com uma nova palavra: (ou digite 'sair' para sair) ")
        if word.lower() == "sair":
            break

        word2 = input("Entre com uma resposta para a nova palavra: ")
        data["intents"].append({
            "tag": "new_intent",
            "patterns": [word],
            "responses": [word2]
        })

    with open("storedata.json", "w") as outfile:
        json.dump(data, outfile, indent=2)

def chat():
    try:
        with open("merged.json") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Nenhum arquivo de dados encontrado. Por favor, execute a função 'merge_json_files' para mesclar os arquivos JSON.")
        return

    try:
        with open("data.pickle", "rb") as file:
            words, labels, training, output = pickle.load(file)
    except FileNotFoundError:
        words, labels, training, output = preprocess_data(data)
        with open("data.pickle", "wb") as file:
            pickle.dump((words, labels, training, output), file)

    try:
        model = tflearn.DNN(net)
        model.load("model.tflearn")
    except (FileNotFoundError, tf.errors.NotFoundError):
        train_model(training, output)
        model.save("model.tflearn")

    print("ChatBot: Olá! Estou aqui para ajudar. Digite 'sair' para sair.")

    while True:
        inp = input("Você: ")
        if inp.lower() == "sair":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        for intent in data["intents"]:
            if intent["tag"] == tag:
                print("ChatBot:", random.choice(intent["responses"]))