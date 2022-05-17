import operator
from collections import defaultdict, Counter
from mynlplib.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Deliverable 3.3
def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars=[]
    ix = 0
    #print(prev_scores[0])
    for cur_tag in list(all_tags):
        c = cur_tag_scores[tag_to_ix[cur_tag]]
        if c == -np.inf:
            bptrs.append(tag_to_ix[START_TAG])
            viterbivars.append(-np.inf)
            continue
        m = -np.inf
        for prev_tag in list(all_tags):
            t = transition_scores[tag_to_ix[cur_tag]][tag_to_ix[prev_tag]]
            p = prev_scores[0][tag_to_ix[prev_tag]]
            #print(c.shape, p.shape, t.shape)
            if p == -np.inf:
                continue
            if t+p+c>m:
                m = t+p+c
                ix = tag_to_ix[prev_tag]
        
        bptrs.append(ix)
        viterbivars.append(m)
        
    viterbivars = Variable(torch.FloatTensor(viterbivars))
    return viterbivars, bptrs 


# Deliverable 3.4
def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    
    Hint: Pay attention to the dimension of cur_tag_scores. It's slightly different from the one in viterbi_step().
    """
    
    ix_to_tag = { v:k for k,v in tag_to_ix.items() }
    
    # setting all the initial score to START_TAG
    # remember that END_TAG is in all_tags
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_ptrs = []
    bptrs = []
    all_scores = []
    path_score = 0.0
    best_path = []
    for m in range(len(cur_tag_scores)):
        scores,bptrs = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores.view(1,-1))
        #print(scores, bptrs)
        all_scores.append(scores)
        prev_scores = scores
        whole_ptrs.append(bptrs)
        
        #raise NotImplementedError

        
   
    # after you finish calculating the tags for all the words: don't forget to calculate the scores for the END_TAG
    m = -np.inf
    ix = 0
    end_vec = np.full(len(all_tags), -np.inf)
    for tag in list(all_tags):
        t = transition_scores[tag_to_ix[END_TAG]][tag_to_ix[tag]]
        p = prev_scores.view(1,-1)[0][tag_to_ix[tag]]
        if p == -np.inf:
            #bptrs.append(ix)
            continue
        if t + p > m:
            m = t + p
            ix = tag_to_ix[tag]
    end_vec[tag_to_ix[END_TAG]] = m
    bptrs = [0] * len(all_tags)
    bptrs[tag_to_ix[END_TAG]] = ix
    whole_ptrs.append(bptrs)
    all_scores.append(Variable(torch.FloatTensor(end_vec)))  
    
    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    path_score = Variable(torch.Tensor(1,1))
    path_score[0] = end_vec[tag_to_ix[END_TAG]]
    cur_tag = END_TAG
    for i in range(len(all_scores)-1,0,-1):
        m = -np.inf
        t = ""
        #cur_tag = ix_to_tag[np.argmax(all_scores[i].detach().numpy())]
        
        for j,score in enumerate(all_scores[i]):
            if score == -np.inf:
                continue
            tag = ix_to_tag[whole_ptrs[i][j]]
            #print(tag,cur_tag)
            #print(transition_scores[tag_to_ix[cur_tag]][tag_to_ix[tag]])
            
            if (score + transition_scores[tag_to_ix[cur_tag]][tag_to_ix[tag]]) > m:
                m = score + transition_scores[tag_to_ix[cur_tag]][tag_to_ix[tag]]
                t = tag
        best_path.append(t)
        cur_tag = t
    best_path.reverse()    
    
    return path_score, best_path
