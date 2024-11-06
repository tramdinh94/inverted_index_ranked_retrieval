# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:31:30 2024

@author: dthtr
"""

#to check time
import time
start_time = time.time()


import sys
import os 
import re
import heapq

import nltk
from nltk import RegexpTokenizer





###___________PREPROCESSING___________________
###_____INDEXING RULES___________________

#stopwords list (sorted) with id
stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'almost', 'already', 'along','another', 'around', 'also', 'although', 'am', 'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 
             'be', 'because', 'before', 'behind', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'co', 'corp',
             'd', 'do', 'down', 'due', 'during', 'each', 'eg', 'elsewhere', 'etc', 'even', 'ever', 'ex', 'few', 'for', 'from', 'further', 'furthermore', 
             'had', 'have', 'he', 'hence', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however',
             'i', 'ie', 'if', 'in',  'inc',  'spite','into', 'it', 'its', 'itself', 'just','less', 'll', 'ltd' , 'm', 'may', 'me', 'meanwhile', 'might', 'more', 'moreover', 'most', 'my', 'myself', 
             'neither', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 're', 
             's', 'same', 'several','she', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'therefore', 'these', 'they', 
             'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'via', 'we', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 
             'will', 'with', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']         


#a dictionary for stopword and their id 
#(which is their sorted index position)
stopword_id = {sw:i for i, sw in enumerate(stopwords)}




#_______LEMMATISER__________________________

#some special case related to the dataset

pos_lem_mapping = {'NN': 'n',       #just noun - but might have gerund
                   'NNPS':'n',      #noun, proper, plural
                   'NNS':'n',       #noun, common, plural
                   'VB': 'v',
                   'VBP':'v',       #VBP: verb, present tense, not 3rd person singular
                   'VBD': 'v',      #VBD: verb, past tense
                   'VBG':'v',       #VBG: verb, present participle or gerund
                   'VBN':'v',       #VBN: verb, past participle
                   'VBZ':'v',       #VBZ: verb, present tense, 3rd person singular
                   'JJ': 'a',       #just adjective - but might have V-ed
                   'JJR':'a',       #JJR: adjective, comparative
                   'JJS':'a',       #JJS: adjective, superlative
                   'RBS': 'r',      #RBR: adverb, comparative
                   'RBR':'r'}       #RBS: adverb, superlative

lemmatiser = nltk.WordNetLemmatizer()









#FUNCTIONS

def lemmatise_no_tag(w):
    pass



    
    
def retrieve_term_postings_one(w):
    global term_posting_dir
    
    firstchar = w[0]
    filepath = os.path.join(term_posting_dir, firstchar)
    
    found = False
    
    with open(filepath, 'r') as f:
        for line in f:
            cur_indexed_term = line[: line.find(',')] 
            if cur_indexed_term == 'w':
                postings = line.split(sep = ',')[1:]
                found = [int(p) for p in postings]
                break
            elif cur_indexed_term > 'w':
                break
    
    return found
    


#if there are multiple term starting with same char
#assume that word is sorted
def retrieve_term_postings_multiple(word_list):
    global term_posting_dir
    
    result = dict()
    
    firstchar = word_list[0][0]
    filepath = os.path.join(term_posting_dir, firstchar)
    
    #to track word
    i = 0
    curword = word_list[i]
    
    with open(filepath, 'r') as f:
        for line in f:
            cur_indexed_term = line[: line.find(',')] 
            if cur_indexed_term < curword:
                continue
            else: 
                if cur_indexed_term == curword:
                    postings = line.strip().split(sep = ',')[1:]
                    result[curword] = [int(p) for p in postings]
                i += 1
                if i < len(word_list):
                    curword = word_list[i]
                else: 
                    break
    return result



    

def retrieve_sw_postings(word_list):
    global sw_posting_dir
    pass





def retrieve_postings(sorted_query):
    pass




def correct_spelling(word_list):
    pass
  

#results will naturally be ordered by docID already
def common_docs(all_posting_lists):   
    results = []
    #number of query terms
    n = len(all_posting_lists)
    pcount = 0
    last = None
    
    for p in heapq.merge(*all_posting_lists):
        if p != last:
            if pcount == n: 
                results.append(p)
            if pcount > n:
                raise Exception(f'something wrong with index, docID {p} appear more than no term')
            pcount = 0
            last = p
        else: 
            pcount += 1
    
    return results
                



def retrieve_positions(docID, w):
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
                
                


#calculate proximity distance between 2 terms in a doc
#p1 is position-list of word 1
#p2 is position-list of word 2
def proximity_distance(doc, p1, p2):
    pass
    


#query must not be sorted
def min_sum_distance(doc, query):
    wp = {w: retrieve_positions(docID, w) for w in query}
    #do some calculation
    pass














#______MAIN__________

#___1___PATH_________

# =============================================================================
# index_dir = sys.argv[1]
# with open(os.path.join(index_dir, 'raw_documents.txt'), 'r') as f:
#     doc_dir = f.readline().strip()
# =============================================================================
    

doc_dir = 'data'
index_dir = 'my_index'

#to store the end position of each doc's line
doc_line_pos_dir = os.path.join(index_dir, "doc_line_endposition")

#specific position of a term/stopword in each doc
doc_term_position_dir = os.path.join(index_dir, "doc_term_position")
doc_sw_position_dir = os.path.join(index_dir, "doc_sw_position")

#postings
term_posting_dir = os.path.join(index_dir, "term_doc")
sw_posting_dir = os.path.join(index_dir, "sw_doc")
     








def process_query(query, print_line):
    
    posting_lists = retrieve_postings(sorted(query))
    
    #if some terms are not found - have to do correct spelling
    # put the results back to posting_lists
    
    not_found =  set(query) - set(posting_lists.keys()) 
    if not_found:
        pass
        
    found_doc = common_docs([posting_lists[w] for w in posting_lists ])
        
    doc_distance = {docID: distance(docID, query) for docID in found_doc}  
    
    
    pass














while True:
    query = input()
    query = query.strip().lower().split()
    if query[0] == '>':
        query = query[1:]
        print_line = True
    else:
        print_line = False
    
    #no sorting query
    query = [lemmatise_no_tag(w) for w in query]    
    
    
    process_query(query, print_line)
    
