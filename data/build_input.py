import numpy as np
import cPickle

def get_idx_from_sent(padding_char,sent, word_idx_map, max_l=51, k=300, filter_h=3):
    """
    Transforms sentence into a list of indices. Padding with zeroes.
    """
    x = []
    pad = filter_h
    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    for i in xrange(pad):
        x.append(word_idx_map[padding_char])

    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def padding(tags,max_l,filter_h=3):
    """
    Transforms tags into a two dimention matrix, where each column correponding to a tag. Padding with zeroes.
    """
    x = np.zeros((max_l+2*filter_h,7))
    tag_index = np.zeros(max_l+2*filter_h)
    for i in xrange(filter_h):
        x[i,0] =1
    index =0
    for word in tags:
        x[filter_h+index,word] =1
        tag_index[filter_h+index] = word
        index+=1
    for i in xrange(filter_h+index,2*filter_h+max_l):
        x[i,0] =1
    return x,tag_index

def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def create_class_weight(labels,mu):
    n_softmax = labels.shape[-1]
    counts = np.zeros(n_softmax, dtype='int32')
    for softmax_index in labels:
        softmax_index = np.asarray(softmax_index)
        for i in range(n_softmax):
            counts[i] = counts[i] + np.count_nonzero(softmax_index==i)

    labels_dict = counterList2Dict(list(enumerate(counts, 0)))

    total = np.sum(labels_dict.values())
    class_weight = dict()

    for key, item in labels_dict.items():
        if not item == 0:
            score = mu * total/float(item)
            class_weight[key] = score if score > 1.0 else 1.0
        else:
            class_weight[key] = 40.0

    return class_weight

def get_sample_weights_multiclass(labels,mu1):
    """
    get samples weights for each sample tags, used in objective functions
    :param labels: a list of list of tag index
    :param mu1: weights parameters
    :return: a list of list weights
    """
    class_weight = create_class_weight(labels,mu=mu1)
    samples_weights = list()
    for instance in labels:
        sample_weights = [class_weight[category] for category in instance]
        samples_weights.append(sample_weights)
    return samples_weights

def make_idx_data(W,revs, word_idx_map,pos_idx_map,chunk_tags_idx_map, max_l=124, k=300):
    """
    Transforms sentences, pos-tag, syntactic chunk tag into a 2-d matrix.
    """
    train, dev, test = [], [], []
    train_tag, dev_tag, test_tag = [], [], []
    train_text, dev_text, test_text = [], [], []
    train_fea2,dev_fea2,test_fea2 = [],[],[]
    train_fea3,dev_fea3,test_fea3 = [],[],[]

    train_weights =[]


    for i, rev in enumerate(revs):

        sent = get_idx_from_sent("\n",rev["text"], word_idx_map, max_l, k,filter_h=3)   #### sent: a list of word index

        ner_tag, index = padding(rev["y"],max_l,filter_h =3)  ##### the tag of the sentences, nertag is 2-d matrix, each column is one-hot vector, index is a list of tag index #####

        pos_tags = get_idx_from_sent("eof",rev["pos_tags"], pos_idx_map, max_l, k,filter_h=3) ###pos_tags: a list of pos-tag-index

        syntactic_chunk_tags = get_idx_from_sent("eof",rev["syntactic_chunk_tags"], chunk_tags_idx_map, max_l, k,filter_h=3) ###syntactic_chunk_tags: a list of syntactic chunk tag index


        if rev["split"] == 0:    #### training data
            train.append(sent)
            train_fea2.append(pos_tags)
            train_fea3.append(syntactic_chunk_tags)
            train_tag.append(ner_tag)
            train_weights.append(index)
            train_text.append(rev["text"])
        elif rev["split"] == 1:  #### dev data
            dev.append(sent)
            dev_text.append(rev["text"])
            dev_fea2.append(pos_tags)
            dev_tag.append(ner_tag)
            dev_fea3.append(syntactic_chunk_tags)
        else:      #### test data
            test.append(sent)
            test_text.append(rev["text"])
            test_fea2.append(pos_tags)
            test_tag.append(ner_tag)
            test_fea3.append(syntactic_chunk_tags)

    train = np.asarray(train, dtype="int")
    dev = np.asarray(dev, dtype="int")
    test = np.asarray(test, dtype="int")

    train_fea2 = np.asarray(train_fea2, dtype="int")
    dev_fea2 = np.asarray(dev_fea2, dtype="int")
    test_fea2 = np.asarray(test_fea2, dtype="int")

    train_fea3 = np.asarray(train_fea3, dtype="int")
    dev_fea3 = np.asarray(dev_fea3, dtype="int")
    test_fea3 = np.asarray(test_fea3, dtype="int")

    train_tag  = np.asarray(train_tag, dtype="int")
    dev_tag  = np.asarray(dev_tag, dtype="int")
    test_tag  = np.asarray(test_tag, dtype="int")

    train_weights = get_sample_weights_multiclass(np.asarray(train_weights), mu1=0.1)

    cPickle.dump([train, dev, test, train_fea2, dev_fea2, test_fea2, train_fea3, dev_fea3, test_fea3,train_tag,dev_tag,test_tag,train_weights,W], open("./colon_input.p", "wb"))



print "loading data...",
x = cPickle.load(open("colln.p","rb"))
revs, W, word_idx_map, vocab,pos_idx_map,chunk_tags_idx_map = x[0], x[1], x[2], x[3], x[4], x[5]
print "data loaded!"
make_idx_data(W,revs, word_idx_map,pos_idx_map,chunk_tags_idx_map, max_l=124, k=300)
print "input created!"