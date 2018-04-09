import torch
import numpy as np
import os
from torch import nn, autograd, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
import json
from collections import Counter
import pickle
import pandas as pd
import re
import codecs
import argparse
import sys
import timeit
from model import CharBasedRNN
from model import Lang

# max sentence length
MAX_SEQ_LEN = 80


# loading an ADM NER file and generates training/eval tensor instances
def load_file(filepath, lang, train=True):
    data = json.load(open(filepath))
    text = data["data"]
    uuid = data["documentMetadata"]["uuid"][0]
    entities = {}
    # reading all entities from ADM file
    for m in data["attributes"]["entities"]["items"]:
        start_tag_id = lang.get_tag_id("B-" + m["mentions"][0]["old-entity-type"])
        inter_tag_id = lang.get_tag_id("I-" + m["mentions"][0]["old-entity-type"])
        start = m["mentions"][0]["startOffset"]
        end = m["mentions"][0]["endOffset"]
        entities[start] = [start, end, start_tag_id, inter_tag_id]
    output_vec = []
    input_vec = []
    curr_tag = None

    # building one long annotated sequence
    for i in range(0, len(text)):
        if train:
            input_vec.append(lang.get_char_id(text[i]))
        else:
            if text[i] in lang.char2id:
                input_vec.append(lang.char2id[text[i]])
            else:
                input_vec.append(0)
        if curr_tag is not None:
            if i >= curr_tag[1]:
                curr_tag = None
                output_vec.append(lang.get_tag_id('O'))
            else:
                output_vec.append(curr_tag[3])
        elif i in entities:
            curr_tag = entities[i]
            output_vec.append(curr_tag[2])
        else:
            output_vec.append(lang.get_tag_id('O'))

    # break the long annotated sequence into sentence instances
    instances = []
    instance_input = []
    instance_output = []
    for i in range(0, len(input_vec)):
        instance_input.append(input_vec[i])
        instance_output.append(output_vec[i])
        if lang.id2char[input_vec[i]] == "." or i == len(input_vec) - 1 or len(instance_input) >= MAX_SEQ_LEN:
            size = len(instance_input)
            instances.append((instance_input, instance_output, size))
            instance_input = []
            instance_output = []

    return uuid, text, instances

# processing a full dataset folder of ADM files and returns a collection
# of all documents
def build_dataset(folderpath, lang):
    data = []
    files = [f for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))]
    for f in files:
        full_path = os.path.join(folderpath, f)
        print "processing", full_path
        _, _, instances = load_file(full_path, lang)
        for i in instances:
            data.append(i)
    return data

# a simple numpy softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# serialize the given model and vocab files into two individual files,
# named with file_path_prefix as a prefix
def serialize(model, lang, file_path_prefix):
    torch.save(model.state_dict(), file_path_prefix + ".model")
    pickle.dump(lang, open(file_path_prefix + ".lang.p", "wb"))

# formating results as ADM for serialization
def prepare_adm(lang, text, tags, uuid):
    data = {"version": "1.1.0",
            "data": text,
            "attributes": {"entities": {"type": "list", "itemType": "entities", "items": []}}}
    start = -1
    tag_type = []
    last_tag = -1
    for t in range(0, len(tags)):
        if start == -1 and tags[t] != 0:
            start = t
            tag_type = [tags[t]]
        elif start != -1 and (tags[t] == 0 or t == len(tags) - 1):
            end = t
            tag_counter = Counter(tag_type)
            # print tag_counter
            tag = tag_counter.most_common(2)
            if len(tag) == 2:
                tag = max(tag[0][0], tag[1][0])
            elif len(tag) == 1:
                tag = tag[0][0]
                if tag % 2 == 1:
                    tag = tag + 1
            # print tag
            m = {"mentions": [{
                "text": text[start:end],
                "old-entity-type": lang.id2tag[tag],
                "startOffset": start,
                "endOffset": end
            }],
                "type": lang.id2tag[tag]}
            start = -1
            tag_type = []
            last_tag = -1
            data["attributes"]["entities"]["items"].append(m)
        else:
            tag_type.append(tags[t])
    data["documentMetadata"] = {"uuid": [uuid]}
    return data
