from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from mynlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


# Deliverable 4.2
def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    
    weights = defaultdict(float)
    
    all_tags = list(trans_counts.keys())+ [END_TAG]
    
    raise NotImplementedError
    
    
    
    return weights


# Deliverable 3.2
def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    for w,value in nb_weights.items():
        if w[1] == '**OFFSET**':
            continue
        tag = w[0]
        word = w[1]
        emission_probs[word_to_ix[word]][tag_to_ix[tag]]=value
    sti = tag_to_ix[START_TAG]
    eti = tag_to_ix[END_TAG]
    for j in range(len(vocab)):
        emission_probs[j][sti] = -np.inf
        emission_probs[j][eti] = -np.inf
        
    for t,value in hmm_trans_weights.items():
        t1 = t[0]
        t2 = t[1]
        if t1 == '--START--':
            tag_transition_probs[tag_to_ix[t1]][tag_to_ix[t2]]= -np.inf
        elif t2 == '--END--':
            tag_transition_probs[tag_to_ix[t1]][tag_to_ix[t2]]= -np.inf
        else:
            tag_transition_probs[tag_to_ix[t1]][tag_to_ix[t2]]=value
        
    #raise NotImplementedError
    
    
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
