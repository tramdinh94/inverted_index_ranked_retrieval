# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:31:30 2024

@author: dthtr
"""

#to check time



import sys
import os 
import re
import heapq
import nltk 
import itertools
import fileinput

from nltk import RegexpTokenizer
from nltk import WordNetLemmatizer

#my own file
from spelling_correction import spelling_correction
from special_vocab import stopwords





###___________PREPROCESSING___________________
###_____INDEXING RULES___________________


#a dictionary for stopword and their id 
#(which is their sorted index position)

#_______LEMMATISER__________________________

#some special case related to the dataset


wnlemmatise = WordNetLemmatizer().lemmatize



def loading_vocab(vocab_dir, vocab_type):
    if vocab_type == 'alphabet':
        firstchars = 'abcdefghijklmnopqrstuvwxyz' 
    
    elif vocab_type == 'digits':
        firstchars = '0123456789'

    vocab = {char:[] for char in firstchars }
    
    all_files = os.listdir(vocab_dir)
    loading_files = [fn for fn in all_files if fn in vocab]
    
    for file in loading_files:
        path = os.path.join(vocab_dir, file)
        d = dict()
            
        with open(path, 'r') as f:
            for line in f:
                item = line.strip().split(sep = ',')
                d[item[0]] = int(item[1])
            
        vocab[file] = d
    return vocab





#FUNCTIONS

def word_preprocess(w):
    if w[0] in '0123456789':
        return w.replace(',', '')
    
    for tag in 'nvar':
        lem = wnlemmatise(w, tag)
        if lem != w:
            return lem
    return w


def query_correction(query):
    global sumf    
    global vocab_alpha
    L = []
    
    multiple = False
    
    for q in query:
        
        if q[0] in '0123456789':
            L.append(q)
        elif q in stopwords:
            L.append(q)
        
        elif q not in vocab_alpha[q[0]]:

            new = spelling_correction(q, vocab_alpha, sumf, obvious = True)
            L.append(new)
            if isinstance(new, list):
                multiple = True
                
        else:
            candidates = spelling_correction(q, vocab_alpha, sumf, obvious = False)
            if not candidates:
                L.append(q)
            else:
                candidates.append(q)
                L.append(candidates)
                multiple = True
 
    
    return L, multiple
                
     


def retrieve_postings(w):
    global term_posting_dir
    global sw_posting_dir
    found = False
    
    if w in stopwords:
        filepath = os.path.join(sw_posting_dir, "1")
    else: 
        firstchar = w[0]
        filepath = os.path.join(term_posting_dir, firstchar)
        

    with open(filepath, 'r') as f:
        for line in f:
            cur_indexed_term = line[: line.find(',')] 
            if cur_indexed_term == w:
                postings = line.split(sep = ',')[1:]
                found = [int(p) for p in postings]
    
    
    return found    
    




#results will naturally be ordered by docID already
#could return empty list
#have to deal with this
def common_docs(single_query):
    
    if len(single_query) == 1:
        return retrieve_postings(single_query[0])
    
    all_posting_lists = [retrieve_postings(w) for w in single_query] 
    
    results = []
    #number of query terms
    n = len(single_query)
    pcount = 0
    last = None
    
    for p in heapq.merge(*all_posting_lists):
       
        if p == last:
            pcount += 1
            if pcount == n-1: 
                results.append(p)
        
            if pcount > n:
                raise Exception(f'something wrong with index, docID {p} appear more than no term')
                return False
            
        else: 
            pcount = 0 #reset counting
            last = p
    
    return results
                



def retrieve_positions(w, docID):
    global doc_term_position_dir
    global doc_sw_position_dir
        
    if w in stopwords:
        folder = doc_sw_position_dir
    else:
        folder = doc_term_position_dir
        
    filepath = os.path.join(folder, str(docID))

    with open(filepath, 'r') as f:
        for line in f:
            cur_indexed_word = line[: line.find(',')] 
            if cur_indexed_word == w:
                positions = line.strip().split(sep = ',')[1:]
                return [int(p) for p in positions]
                
                


#calculate proximity distance 
#and number of right orders
def min_sum_distance(docID, query):
    
    n = len(query)
    
    LP = [retrieve_positions(w, docID) for w in query]
    
    if n == 1:
        return 0, 0, [LP[0][0]]
    
    position_combinations = list(itertools.product(*LP))
    
    
    min_dis =  65535
    
    for c in position_combinations:
        sum_dis = 0
        ordered = 0
        for i in range(0, n-1):
            d = c[i+1] - c[i]
            sum_dis += abs(d)
            if d > 0:
                ordered += 1
        #update everything if find new min
        if sum_dis < min_dis:
            min_dis = sum_dis
            min_c = c
            min_ordered = ordered 
        
        #if cannot improve distance
        #but can increase the number of ordered pairs
        #or the combination is at higher lines
        if sum_dis == min_dis:
            if (ordered > min_ordered) or (ordered == min_ordered and min(c) < min(min_c)):
                min_c = c
                min_ordered = ordered
        
            
    return min_dis, min_ordered, min_c




def processed_query(original_query):
    
    query_updated, multiple = query_correction(original_query)
    
    if not multiple:
        query_combination = [query_updated]
    else:
        query_combination = [x if isinstance(x, list) else [x] for x in query_updated]
        query_combination = list(itertools.product(*query_combination))
        
    
    ranked_docs = []
    
    for query in query_combination:
        docs = common_docs(query)

        
        if docs:  #check if not empty
            for docID in docs:
                min_dis, ordered, c = min_sum_distance(docID, query)
                ranked_docs.append( (docID, min_dis, ordered, c))
            
    
    ranked_docs.sort(key = lambda x: (x[1], - x[2], x[0] ))
    
    return ranked_docs
        
        
#print a doc 
#given c = the best combination of positions    
def print_lines(docID, position_combination):
    global doc_line_dir
    
    Lines = []
    
 
    c = sorted(position_combination, reverse=True)
    path = os.path.join(doc_line_dir, str(docID))
    
    
    with open(path, 'r') as f:
        for s in f:
            i = s.find(',')
            endpos = int(s[:i])
            if c[-1] <= endpos:
                Lines.append(s[i+1:])
                while c and c[-1] <= endpos:
                    c.pop()
            if len(c) == 0:
                break
            
    for line in Lines:
        print(line,  end='')
                    
                
                


def query_results(input_string):
    query = input_string.strip().lower().split()

    if query[0] == '>':
        query = query[1:]
        pain_in_the_ass = True
    else:
        pain_in_the_ass = False
    
    #no sorting query
    query = [word_preprocess(w) for w in query]    
    
    
    ranked_docs = processed_query(query)
    printed_docs = set()
    for item in ranked_docs:
        if item[0] not in printed_docs:
            if not pain_in_the_ass:
                print(item[0])
            else:
                print("> " + str(item[0]))
                print_lines(item[0], item[-1])
            printed_docs.add(item[0])
            
    
    

            
        


# =============================================================================
#____MAIN____________

#___1___PATH_________

index_dir = sys.argv[1]




#specific position of a term/stopword in each doc
doc_term_position_dir = os.path.join(index_dir, "doc_term_position")
doc_sw_position_dir = os.path.join(index_dir, "doc_sw_position")

#postings
term_posting_dir = os.path.join(index_dir, "term_doc")
sw_posting_dir = os.path.join(index_dir, "sw_doc")
     
#vocab
vocab_dir = os.path.join(index_dir, 'vocabulary')

#line
doc_line_dir = os.path.join(index_dir, 'doc_line_endpos')

vocab_alpha = loading_vocab(vocab_dir, 'alphabet')




with open(os.path.join(index_dir, 'reference.txt'), 'r') as f:
    L = [line.strip() for line in f]
    doc_dir = L[0]
    sumf = int(L[1])
   



if len(sys.argv) > 2:
    with open(sys.argv[-1], 'r') as input_queries:
        for line in input_queries:
            query_results(line)  
else:
    input_queries = sys.stdin
    for line in input_queries:
        query_results(line)
    

    

