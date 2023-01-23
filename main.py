import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import ast
import random
import pandas as pd

characters = { #dictionary of all characters and there resulting keycode

    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,

    ".": 27,
    ",": 28,
    "?": 29,
    "!": 30,
    ":": 31,
    ";": 32,
    "'": 33,
    "[": 34,
    "]": 35,
    "(": 36,
    ")": 37,

    "0": 38,
    "1": 39,
    "2": 40,
    "3": 41,
    "4": 42,
    "5": 43,
    "6": 44,
    "7": 45,
    "8": 46,
    "9": 47,

    " ": 48,
    "—": 49,
    "_": 50,
    "’": 51,
    "-": 52
}

with open('CommunistManifesto.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    f.close()


def checkline(line):
    succeed = True
    for ch in line:
        if ch not in characters:
            succeed = False
    return succeed

def converttobinary(num):
    binary_num = str(bin(num)[2:])
    while len(binary_num) < 6: #converts number into a 6bit binary number
        binary_num = "0" + binary_num
    return binary_num
def remove_lines(value):
    return ' '.join(value.splitlines())

data = []
label_data = []
save_num_list = []
data_length = 100
gatherData = False
chunks = 9
chunkSize = 3100
if gatherData:
    for x in range(30000):  # for every character in the communist manifesto 72500
        print(x)
        fail = False
        binary_str = []  # stores the binary data of an input
        following_binary_str = []  # stores the binary data of the following output

        textpointer = x
        textPiece = text[textpointer:textpointer+data_length] # takes a piece from the text
        textPiece = remove_lines(textPiece) # removes lines from that piece for ease of use
        if True:  # checks that the line doesn't cut through a word, optional textPiece[0] == " "
            textPiece = textPiece.lower()
            followingLine = text[textpointer+data_length:textpointer+data_length+1]  # following output letter

            if checkline(textPiece):
                for ch in textPiece:
                    binary_num = converttobinary(characters[ch])  # converts each individual ch to binary
                    for n in binary_num:
                        binary_str.append(n)
            else:
                fail = True

            if checkline(followingLine):
                for ch in followingLine:
                    save_num = characters[ch] # use for predicting a single character
            else:
                fail = True

            if fail == False and binary_str not in data and len(binary_str) == data_length*6:
                data.append(binary_str)
                save_num_list.append(save_num)

    with open('Dataset.txt', 'w', encoding='utf-8') as f:
        f.write(str(data))
        f.close()
    with open('Datalabel.txt', 'w', encoding='utf-8') as f:
        f.write(str(save_num_list))
        f.close()
    print(len(data))
    exit()

train_images = []
train_labels = []
with open('Dataset.txt', 'r', encoding='utf-8') as f:
    for x in range(chunks):
        lines = [i + (chunkSize * x) for i in range(chunkSize)]

        for pos, l_num in enumerate(f):
            # check if the line number is specified in the lines to read array
            if pos in lines:
                # print the required line number
                linedata = ast.literal_eval(l_num)
        train_images += linedata
    f.close()

with open('Datalabel.txt', 'r', encoding='utf-8') as f:
    for x in range(chunks):
        lines = [i + (chunkSize * x) for i in range(chunkSize)]

        for pos, l_num in enumerate(f):
            # check if the line number is specified in the lines to read array
            if pos in lines:
                # print the required line number
                linedata = ast.literal_eval(l_num)
        train_labels += linedata
    f.close()


print(len(train_images))
print(len(train_labels))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = np.array(train_images, dtype=float)
train_labels = np.array(train_labels, dtype=float)


model = keras.Sequential([
    keras.layers.Dense(data_length*6),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(275, activation="relu"),
    keras.layers.Dense(225, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(90, activation="relu"),
    keras.layers.Dense(75, activation="relu"),
    keras.layers.Dense(53, activation="sigmoid")
    ])
model.compile(optimizer="Adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=100, use_multiprocessing=False)

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("test accuracy: ", test_acc)


while True:
    Full_binary_query_list = []
    binary_query_list = []
    message = []
    added_binary_letter = []
    number_query_list = []

    query = str(input("what is your sentence"))

    if len(query) > data_length:
        query = query[0:data_length]
    while len(query) < data_length:
        query = " " + query
    if checkline(query):
        for ch in query:
            binary_query = converttobinary(characters[ch])
            for n in binary_query:
                binary_query_list.append(n)
    for x in range(1):
        Full_binary_query_list.append(binary_query_list)

    binary_query_list = np.array(Full_binary_query_list)


    for x in range(50):
        if x == 0:
            binary_query_list = np.array(Full_binary_query_list, dtype=float)
        else:
            binary_query_list = np.array(binary_query_list, dtype=float)
            binary_query_list = np.reshape(binary_query_list, (1, data_length*6))

        prediction = model.predict(binary_query_list)

        value = {i for i in characters if characters[i] == np.argmax(prediction[0])}
        added_binary_letter = converttobinary(np.argmax(prediction[0]))
        added_binary_letter_list = []

        for x in added_binary_letter:
            added_binary_letter_list.append(x)

        binary_query_list = np.append(binary_query_list, added_binary_letter_list)
        binary_query_list = np.delete(binary_query_list, [0, 1, 2, 3, 4, 5])


    for x in range(round(len(binary_query_list)/6)):
        starter = 32
        total = 0
        for i, l in enumerate(binary_query_list[x*6:(x+1)*6]):
            if type(l) == np.str_:
                if l == '1.0':
                    total += starter

            elif round(l) == 1:
                total += starter
            starter = starter/2
        number_query_list.append(total)


    for num in number_query_list:
        message.append({i for i in characters if characters[i] == num})
    message = message[0:-2]
    for list in message:
        for ch in list:
            print(ch)







