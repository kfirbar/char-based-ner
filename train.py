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
from eval import *

print("PyTorch Version: ", torch.__version__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_folder", help="the location for the training data")
    parser.add_argument("--eval", help="the location for the eval data")
    parser.add_argument("--eval_results", help="where to put the evaluation results", default=None)
    parser.add_argument("--model", help="the last checkpoint")
    parser.add_argument("--vocab", help="the last checkpoint lang file")
    parser.add_argument("--lang", help="the language we train on (e.g., eng, jpn)")
    parser.add_argument("--hidden_size", help="the hidden size of the network", type=int, default=500)
    parser.add_argument("--embedding_size", help="the character embeddings size of the network", type=int, default=200)
    parser.add_argument("--layers", help="the number of LSTM layers", type=int, default=3)
    parser.add_argument("--epochs", help="the number of epochs to run", type=int, default=1)
    parser.add_argument("--first_epoch_number", help="the number of the first epoch", type=int, default=1)
    parser.add_argument("--batch_size", help="the size of a batch", type=int, default=20)
    parser.add_argument("--print_every", help="print every this number of batches", type=int, default=1)
    parser.add_argument("--sample_every", help="sample every this number of batches", type=int, default=2)
    parser.add_argument("--checkpoint_every", help="checkpoint every this number of batches", type=int, default=2)
    parser.add_argument("--use_gpu", help="run on GPU", type=bool, default=False)

    args = parser.parse_args()

    current_loss = 0
    counter = 0

    lang = None
    if args.vocab is not None:
        print "Loading Vocab from ", args.vocab
        lang = pickle.load(open(args.vocab, "rb"))
    else:
        lang = Lang()
    data = build_dataset(args.train_folder, lang)
    print "Number of training instances:", len(data)
    print "Tags:", lang.tag2id

    model = CharBasedRNN(lang.n_chars, args.hidden_size, args.embedding_size, lang.n_tags, args.layers, args.use_gpu)
    if args.model is not None:
        print "Loading model data from ", args.model
        model.load_state_dict(torch.load(args.model))
    if args.use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # main epoch loop
    for e in range(args.first_epoch_number, args.epochs + args.first_epoch_number):
        # before running epoch, randomize the data
        perm = np.random.permutation(len(data))
        batch_indexes = [perm[i:i + args.batch_size] for i in xrange(0, len(perm), args.batch_size)]

        # main batch loop
        for bi in batch_indexes:
            input_batch = []
            target_batch = []

            counter += 1
            for r in xrange(0, len(bi)):
                pair = data[bi[r]]
                input_batch.append(pair[0])
                target_batch.append(pair[1])

            loss = 0

            # the model changes the order of the batch, to allow variable lengths,
            # therefore perm_idx is the mapping between the original order and the output order
            output, perm_idx = model(input_batch)

            input_batch = [input_batch[i] for i in perm_idx.cpu().numpy()]
            target_batch = [target_batch[i] for i in perm_idx.cpu().numpy()]

            for inp in xrange(0, len(output)):
                short_output = output[inp][0:len(input_batch[inp])]
                if args.use_gpu:
                    loss += criterion(short_output, Variable(torch.LongTensor(target_batch[inp]).cuda()))
                else:
                    loss += criterion(short_output, Variable(torch.LongTensor(target_batch[inp])))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()
            current_loss += loss.data[0] / args.batch_size

            if counter % args.print_every == 0:
                current_loss = current_loss / args.print_every
                print 'Epoch %d, Batch %d Current Loss = %.4f' % (e, counter, current_loss)
                current_loss = 0
            if counter % args.sample_every == 0:
                print "Target:", target_batch[0]
                short_output = output[0][0:len(input_batch[0])].data.cpu().numpy()
                seq_output = []
                for ou in short_output:
                    output_tag_id = np.argmax(softmax(ou))
                    seq_output.append(output_tag_id)
                print "Output:", seq_output
            if counter % args.checkpoint_every == 0:
                serialize(model, lang, args.lang + "-cp-" + str(counter) + "-l-" + str(args.layers) + "-hs-" + str(
                    args.hidden_size) + "-e-" + str(args.embedding_size))

    serialize(model, lang, args.lang + "-cp-final-l-" + str(args.layers) + "-hs-" + str(args.hidden_size) + "-e-" + str(
        args.embedding_size))
    print "Done with training!"

    start_time = timeit.default_timer()

    if args.eval_results is not None:
        print "Starting evaluation..."
        total_length_chars = eval(args.eval, args.eval_results, model, lang)
        elapsed = timeit.default_timer() - start_time
        print "Done with evaluation! it took ", elapsed, "seconds, #characters:", total_length_chars


if __name__ == "__main__":
    main()
