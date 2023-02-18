
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

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

data_length = 80
NN_fill_length = 40

model = keras.models.load_model("best_model.hdf5")

while True:
    Full_binary_query_list = []
    binary_query_list = []
    message = []
    added_binary_letter = []
    number_query_list = []
    formatted_message = []

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


    for x in range(NN_fill_length):
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
    listed_message = []
    for x in message:
        for y in x:
            listed_message.append(y)
    print("".join([str(y) for y in listed_message]))



