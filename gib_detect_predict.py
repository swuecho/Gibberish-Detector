#!/usr/bin/python

import pickle
import gib_detect

model_data = pickle.load(open('gib_model.pki', 'rb'))


def predict(text):
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    return gib_detect.avg_transition_prob(text, model_mat) > threshold


while True:
    text = input("please give me some text\n")
    print(predict(text))
