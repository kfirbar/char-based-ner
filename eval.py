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
from utils import *

print("PyTorch Version: ", torch.__version__)

MAX_SEQ_LEN = 80


def eval_file(filepath, eval_folder_path, model, lang):
    uuid, text, instances = load_file(filepath, lang, train=False)
    seq_output = []
    total_length_chars = 0
    input_batch = []
    target_batch = []
    for i in range(0, len(instances)):
        input_var = instances[i][0]
        total_length_chars += len(input_var)
        input_batch.append(input_var)

    output, perm_idx = model(input_batch)

    orig_output = [None]*len(output)
    ind = 0
    for o in xrange(0, len(perm_idx)):
        orig_output[perm_idx[o]] = output[ind]
        ind += 1

    for inp in xrange(0, len(orig_output)):
        short_output = orig_output[inp][0:len(input_batch[inp])].data.cpu().numpy()
        for ou in short_output:
            output_tag_id = np.argmax(softmax(ou))
            seq_output.append(output_tag_id)

    # print seq_output
    adm_data = prepare_adm(lang, text, seq_output, uuid)
    head, tail = os.path.split(filepath)
    eval_path = os.path.join(eval_folder_path, tail)
    text_file = open(eval_path, "w")
    text_file.write(json.dumps(adm_data, indent=4))
    text_file.close()
    return total_length_chars


def eval(eval_folder_path, results_folder_path, model, lang):
    total_length_chars = 0
    files = [f for f in os.listdir(eval_folder_path) if os.path.isfile(os.path.join(eval_folder_path, f))]
    for f in files:
        full_path = os.path.join(eval_folder_path, f)
        print "processing", full_path
        total_length_chars += eval_file(full_path, results_folder_path, model, lang)
    return total_length_chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="the model file")
    parser.add_argument("lang", help="the vocab file")
    parser.add_argument("eval", help="the eval folder")
    parser.add_argument("eval_results", help="where to put the evaluation results", default=None)
    parser.add_argument("--hidden_size", help="the hidden size of the network", type=int, default=500)
    parser.add_argument("--embedding_size", help="the character embeddings size of the network", type=int, default=200)
    parser.add_argument("--layers", help="the number of LSTM layers", type=int, default=3)
    parser.add_argument("--use_gpu", help="run on GPU", type=bool, default=False)

    args = parser.parse_args()
    lang = pickle.load(open(args.lang, "rb"))

    print "Tags:", lang.tag2id

    model = CharBasedRNN(lang.n_chars, args.hidden_size, args.embedding_size, lang.n_tags, args.layers, args.use_gpu)
    model.load_state_dict(torch.load(args.model))
    if args.use_gpu:
        model = model.cuda()

    start_time = timeit.default_timer()

    print "Starting evaluation..."
    total_length_chars = eval(args.eval, args.eval_results, model, lang)
    elapsed = timeit.default_timer() - start_time
    print "Done with evaluation! it took ", elapsed, "seconds, #characters:", total_length_chars


if __name__ == "__main__":
    main()
