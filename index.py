# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:43:23 2024

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

grammar_term = ["'", ".", ",", "?", "!","'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]

not_index = ["'", ".", "?", "!", ","]





###___________TOKENISER____________
#pattern capture descimal number, number with comma, and number
#capture in order: number with decimal, number (optional comma), 
#gramma contraction, punctuation, abbreviation, term(word and number)

pattern = r'\d+(?:\.\d+)'\
    r'|\d+(?:\,\d{3})*'\
    r'|(?:\'(?:s|ve|ll|m|re|d|t)\b)'\
    r"|[.?!']" \
    r'|(?:[a-z]\.)+'\
    r'|(?:[\w\d]+)'\

my_tokeniser = RegexpTokenizer(pattern)




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



#_______FUNCTION___________________________

#For setting path
def create_dir(parent, child):
    if parent:
        new_dir = os.path.join(parent, child)
    else:
        new_dir = child      
    #create dir if not existing
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    #return the path of new dir    
    return new_dir  




def add_pos(d, k, p, duplicate_p = False):
    #adding posting or position to a dictionary (key = term)
    # k, p = term and individual posting (docID) if dict keys = terms
    # k, P = doc and term position in a doc if dict keys = docIDs       
    if k in d:
        if duplicate_p == False or d[k][-1] != p:  
            d[k].append(p)
    else:
        d[k] = [p]       
   
            
   
#print pos (posting or position) from an item 
#item can be list, set or dictionary
def print_pos(items, path, mode, keys, print_blank = False):
    with open(path, mode) as f:
        for k in keys:
            if k in items:
                line = k if isinstance(k, str) else str(k)
                line += ',' + ','.join([str(p) for p in items[k]]) + '\n' 
                f.write(line)
            elif print_blank: 
                line = '\n'
                f.write(line)
    


def line_tokenise(line):
    #strip line, turn all into lower case, then add "EOF" to mark new line
    s = line.strip().lower() 
    
    #transform all n't to not
    s = re.sub(r"(\b(?:are|could|do|does|did|has|have|is|must|should|would|was|were))n\'t\b", r' \1 not ', s)
    s = re.sub(r"\b(?:ai)n\'t\b", ' are not ', s)
    s = re.sub(r"\b(?:ca)n\'t\b", ' can not ', s)
    s = re.sub(r"\b(?:sha)n\'t\b", ' shall not ', s)
    s = re.sub(r"\b(?:wo)n\'t\b", ' will not ', s)
    
    
    tokens = my_tokeniser.tokenize(s) 
    
    return tokens


def lemmatise_token(tk, tag):
    #if it is singular noun
    if tag not in pos_lem_mapping:
        return tk
    
    #speical case: V-ing/gerun tagged as noun
    if tag == 'NN' and tk.endswith('ing'):
        lem_as_v = lemmatiser.lemmatize(tk, 'v')
        if lem_as_v != tk: 
            return lem_as_v
        else:
            return tk
        
    #special case: special noun in plural form not recognised by lemmatiser
    #but correctly tagged by pos_tag
    elif (tag == 'NNS' or tag == 'NNPS') and tk.endswith('s'):
        lem_as_n = lemmatiser.lemmatize(tk, 'n')
        #if lemmatiser fail to trim it
        #have to trim it manually
        if lem_as_n == tk: 
            return tk[:-1]
        #but if lemmatiser can catch it
        #return the lemmatised word
        else:
            return lem_as_n
            
    #special case: V-past-participle tagged as adjective
    elif tag == 'JJ':
        lem_as_v = lemmatiser.lemmatize(tk, 'v')
        if lem_as_v != tk: 
            return lem_as_v
        else:
            return tk
    
    else:
        return lemmatiser.lemmatize(tk, pos_lem_mapping[tag])
    


def index_term(tk, tag):
    #Check condition on the token
    #if symbol
    if tk in not_index:
        return False
       
    # if token = 's , it can be either verb (index) or possesive (not index)
    elif tk == "'s":
        if tag == 'POS': 
            return False
        else:
            term = 's'
    
    #if token has the ' symbol
    elif tk in ( "'ve", "'ll", "'m", "'re", "'d", "'t"):
        term = tk[1:]
        
    elif '.'in tk: 
        if tag == 'CD':
            return False
        else: 
            term = tk.replace('.','')
        
    #Conditions on tag
    #if token = number
    #decimal - not index
    #other cases: remove comma (if any), and then index
    elif tag == 'CD':
        term = tk.replace(',', '')  
            
    else:
        term = lemmatise_token(tk, tag)
        
    return term





#doc ID is provided as interger
def doc_parsing(docID, doc_path):  
    global stopwords
    global doc_term_position_dir, doc_sw_position_dir
    global posting_sw
    global posting_term
    
    term_pos = dict()
    sw_pos = dict()
    #PARSING LINE BY LINE
    #record the position of last indexing tokens on the line
    with open(doc_path, 'r') as f:
        tokens = [line_tokenise(line) for line in f]
    
    no_line = len(tokens)
    line_endposition = [0 for i in range(0, no_line)]  
    
    #flatten the list of tokens
    tokens = [item[i] for item in tokens for i in range(0,len(item))]
      
    #pos_tag, lemmatise, and index
    tokens_tag = nltk.pos_tag(tokens)
    position = 0
    line = 0
    
    for i, tk in enumerate(tokens_tag):
        #if it is the end of line
        #which is marked by an inserted comma
        #this is because somehow comma does not affect pos_tag accuracy
        if tk[0] == ',':
            if i > 0 and tokens_tag[i-1] != ',': 
                line_endposition[line] = position
            line += 1
            
        else:
            term = index_term(tk[0], tk[1])
            if term: 
                position += 1
                if term in stopwords:
                    sw_id = stopword_id[term]
                    add_pos(sw_pos, term, position)
                    add_pos(posting_sw, sw_id, docID, duplicate_p =True)
                else:
                    add_pos(term_pos, term, position)
                    add_pos(posting_term, term, docID, duplicate_p =True)
            
    sorted_terms = sorted(term_pos.keys())
    

    #print it out
    print_pos(term_pos,  
              path = os.path.join(doc_term_position_dir, str(docID)), 
              mode = 'w',
              keys = sorted_terms)
    
    print_pos(sw_pos,
              path = os.path.join(doc_sw_position_dir, str(docID)),
              mode = 'w',
              keys = stopwords)
       
    return sorted_terms
            
     
        
#parsing all doc and return a heapq merge generator
#vocab  
def doc_to_vocab_posting():
    global posting_sw
    global posting_term
    
    
    vocab = []
    
    #PARSING THROUGH ALL DOC
    Docs = os.listdir(doc_dir)
    
    #to parse doc in ordered so the docs are added in order
    Docs.sort(key = lambda x: int(x))
    
    for i, docID in enumerate(Docs): 
        #term_doc block
        doc_path = os.path.join(doc_dir, docID)
        #adding sorted list of (term, doc) to block
        vocab.append(doc_parsing(int(docID), doc_path) )           

    return heapq.merge(*vocab)



#take a vocab generator  from previous function          
def merge_sort_group_print(vocab): 
    global posting_term
    global term_posting_dir
    #how posting lists will be print 
    #according to the first character of the terms
    firstchar = ['0123456789','ab','cd','efgh','ijkl','mnopq','rs','tuvwxyz']
    #block index
    i = 0
    curblock = []
    last = None
    for term in vocab:
        if term != last:  # if it is not a duplicate
            if term[0] in firstchar[i]: 
                curblock.append(term)
            else: 
                #if the next term's first char
                #belong to other group
                #print the current group
                if curblock: #if not empty
                    print_pos(items = posting_term, 
                              path = os.path.join(term_posting_dir, firstchar[i]), 
                              mode = 'w', 
                              keys = curblock)                
                curblock = []
                
                #find the next first char group
                while term[0] not in firstchar[i]:
                    i+=1
                #and add the current term to it    
                curblock.append(term)
            
            last = term
    if curblock:
        print_pos(items = posting_term, 
                  path = os.path.join(term_posting_dir, firstchar[i]), 
                  mode = 'w', 
                  keys = curblock)   



#___________MAIN BODY______________________

#___1___GETTING PROVIDED PATH FOR DATA AND INDEX
#doc_dir = sys.argv[1]
#index_folder = sys.argv[2]
doc_dir = 'data'
index_folder = 'my_index'

index_block_size = 300






#___2___SETTING UP PATHS FOR INDEX FOLDER AND SUBFOLDERS 
#FOR THIS PROGRAM: MAIN INDEX FOLDER ONLY HAVE 1 FILE (contain path to raw data)
#AND OTHER SUBFOLDERS
#THESE SUBFOLDERS CANNOT CONTAIN ANY SUBFOLDERS, only files with integer name

index_dir = create_dir(parent = False, child = index_folder)

#to store the end position of each doc's line
doc_line_pos_dir = create_dir(index_dir, "doc_line_endposition")

#to store the specific position of a term/stopword in each doc
doc_term_position_dir = create_dir(index_dir, "doc_term_position")
doc_sw_position_dir = create_dir(index_dir, "doc_sw_position")


#POSTING FOR MAIN TERMS AND STOPWORDS
term_posting_dir = create_dir(index_dir, "term_doc")
sw_posting_dir = create_dir(index_dir, "sw_doc")

      

#first, write the path of doc
with open(os.path.join(index_dir, "raw_documents.txt"), 'w') as f:
          f.write(doc_dir + '\n')
          


#__3___BUILDING INVERTED INDEX STEP BY STEP

#__3a__extract data (parsing doc one by one, flush results out to files)

#just because 2 non-nested function need to access it
posting_sw = dict()
posting_term = dict()

#vocab (main terms) - generator
vocab = doc_to_vocab_posting()

#print terms-postings, grouped by first character
merge_sort_group_print(vocab)

#print stopwords-postings
print_pos(posting_sw, 
          path = os.path.join(sw_posting_dir, '1'), 
          mode = 'w', 
          keys = range(0, len(stopwords)), 
          print_blank=True)


#check time            
print("running time to this point is %s seconds" % (time.time() - start_time))        
        
 
      


      