import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from random import randint
np.random.seed(20171025)


def build_tag_dict(path):
    data = open(path).read()
    tag_dict = defaultdict(float)
    index =0
    for line in data.splitlines():
        tag_dict[line] = index
        index+=1
    return tag_dict




def build_data(data_folder):
    """
     preprocess training, dev, and test file, return lists of words, pos tags, syntactic chunks, frequency dictionaries
    """
    revs = []
    [train_file,dev_file,test_file] = data_folder
    vocab = defaultdict(float)
    pos_tags_vocab = defaultdict(float)
    syntactic_chunk_tags_vocab = defaultdict(float)
    tag_dict = build_tag_dict("./raw/label.txt")
    with open(train_file, "rb") as f:
        start_doc = True
        for line in f:
            if line.startswith("-DOCSTART-"):
                start_doc = True
                continue
            elif len(line)==1 and start_doc:
                start_doc = False
                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            elif len(line)==1 and not start_doc:
                datum  = {"y":tags,                   ### tags: the tags of one sentence
                          "text": text,               ### text, a list of word of one sentence
                          "pos_tags":pos_tags,        ### pos_tag, a list of pos tag for one sentence
                          "syntactic_chunk_tags":syntactic_chunk_tags, ### syntactic_chunk_tags, a list of syntactic chunk tag for one sentence
                          "num_words": len(text),    ### the length of one text
                          "split": 0}             ### 0: training file, 1 : dev file, 2: test_file
                revs.append(datum)
                start_doc = False

                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            else:
                line = line.strip()
                input = line.split(" ")
                tags.append(int(tag_dict[input[3]]))
                text.append(input[0])
                pos_tags.append(input[1])
                syntactic_chunk_tags.append(input[2])
                vocab[input[0]] += 1            ############  key: word; value : frequency of the word
                pos_tags_vocab[input[1]] +=1     ############  key: pos-tag; value : frequency of the pos-tag
                syntactic_chunk_tags_vocab[input[2]] +=1     ###### key: syntactic-chunk-tag; value : frequency of the syntactic-chunk-tag
                start_doc = False

    #print revs[-1]['text']

    with open(dev_file, "rb") as f:
        start_doc = True
        for line in f:
            if line.startswith("-DOCSTART-"):
                start_doc = True
                continue
            elif len(line)==1 and start_doc:
                start_doc = False
                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            elif len(line)==1 and not start_doc:
                datum  = {"y":tags,
                          "text": text,
                          "pos_tags":pos_tags,
                          "syntactic_chunk_tags":syntactic_chunk_tags,
                          "num_words": len(text),
                          "split": 1}
                revs.append(datum)
                start_doc = False
                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            else:
                line = line.strip()
                input = line.split(" ")
                tags.append(int(tag_dict[input[3]]))
                text.append(input[0])
                pos_tags.append(input[1])
                syntactic_chunk_tags.append(input[2])
                vocab[input[0]] += 1
                pos_tags_vocab[input[1]] +=1
                syntactic_chunk_tags_vocab[input[2]] +=1
                start_doc = False

    with open(test_file, "rb") as f:
        start_doc = True
        for line in f:
            if line.startswith("-DOCSTART-"):
                start_doc = True
                continue
            elif len(line)==1 and start_doc:
                start_doc = False
                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            elif len(line)==1 and not start_doc:
                datum  = {"y":tags,
                          "text": text,
                          "pos_tags":pos_tags,
                          "syntactic_chunk_tags":syntactic_chunk_tags,
                          "num_words": len(text),
                          "split": 2}
                revs.append(datum)
                start_doc = False
                text = list()
                tags = list()
                pos_tags = list()
                syntactic_chunk_tags = list()
            else:
                line = line.strip()
                input = line.split(" ")
                tags.append(int(tag_dict[input[3]]))
                text.append(input[0])
                pos_tags.append(input[1])
                syntactic_chunk_tags.append(input[2])
                vocab[input[0]] += 1
                pos_tags_vocab[input[1]] +=1
                syntactic_chunk_tags_vocab[input[2]] +=1
                start_doc = False

    vocab["\n"] += 1
    pos_tags_vocab["eof"] +=1
    syntactic_chunk_tags_vocab["eof"] +=1
    return revs, vocab,pos_tags_vocab,syntactic_chunk_tags_vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec, and output a embedding matrix
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')    ###create a markup word vector embedding
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def get_idx_map(vocab):
    """
    transform a word-frequency into a word-index dictionary
    :param vocab:  dictionary, key: word; value: frequency
    :return: vword-index vocabulary
    """
    word_idx_map = dict()
    i =1
    for key,item in vocab.iteritems():
        word_idx_map[key] =i
        i+=1
    return word_idx_map


stsa_path = "./raw/"
w2v_file = "./w2v/GoogleNews-vectors-negative300.bin"
#### read input ########
train_data_file = "%s/eng.train" % stsa_path
dev_data_file = "%s/eng.testa" % stsa_path
test_data_file = "%s/eng.testb" % stsa_path
data_folder = [train_data_file, dev_data_file, test_data_file]
print "loading data...",
###### processing datasets #####
revs, vocab,pos_tags_vocab,syntactic_chunk_tags_vocab = build_data(data_folder)
##### maximal length of the sentence, used in markup ########
max_l = np.max(pd.DataFrame(revs)["num_words"])
n_pos = len(pos_tags_vocab) +1
n_chunk_tags = len(syntactic_chunk_tags_vocab) +1
print "data loaded!"
print "number of sentences: " + str(len(revs))
print "vocab size: " + str(len(vocab))
print "pos vocab size: " + str(n_pos)
print "chunk tag vocab size: " + str(n_chunk_tags)
print "max sentence length: " + str(max_l)
print "loading word2vec vectors...",
w2v = load_bin_vec(w2v_file, vocab)
print "word2vec loaded!"
print "num words already in word2vec: " + str(len(w2v))
add_unknown_words(w2v, vocab)
W, word_idx_map = get_W(w2v)
pos_idx_map = get_idx_map(pos_tags_vocab)
chunk_tags_idx_map  = get_idx_map(syntactic_chunk_tags_vocab)

cPickle.dump([revs, W, word_idx_map, vocab,pos_idx_map,chunk_tags_idx_map], open("./colln.p", "wb"))
print "dataset created!"


