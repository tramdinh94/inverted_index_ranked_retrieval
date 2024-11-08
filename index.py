# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:43:23 2024

@author: dthtr
"""


#to check time
#import time
#start_time = time.time()


import sys
import os 
import re
import heapq
import nltk

from nltk import RegexpTokenizer
from nltk import WordNetLemmatizer

#from my own files
from special_vocab import stopwords


###___________PREPROCESSING___________________
###_____INDEXING RULES___________________

#stopwords list (sorted) 


#a dictionary for stopword and their id 
#(which is their sorted index position)
stopword_id = {sw:i for i, sw in enumerate(stopwords)}

grammar_term = ["'", ".", "?", "!","'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]

not_index = ["'", ".", "?", "!"]





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

lemmatiser = WordNetLemmatizer()



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




def add_position(d, k, p):
    #adding posting or position to a dictionary (key = term)
    # k, p = term and individual posting (docID) if dict keys = terms
    # k, P = doc and term position in a doc if dict keys = docIDs       
    if k in d:
        d[k].append(p)
    else:
        d[k] = [p]       
   


def add_posting(d, k, p, word_type):
    #adding posting or position to a dictionary (key = term)
    # k, p = term and individual posting (docID) if dict keys = terms
    # k, P = doc and term position in a doc if dict keys = docIDs      
    if word_type == 'term':  
        if k in d:
            d[k][0] += 1
            if d[k][1][-1] != p:  
                d[k][1].append(p)
        else:
            d[k] = [1, [p]] 
    
    elif word_type == 'sw': #if it is just stopwords - no need to have frequency
        if k not in d:
            d[k] = [p]
        elif d[k][-1] != p:
            d[k] = [p]

            
   
#print pos (posting or position) from an item 
#item can be list, set or dictionary
def print_position(items, path, mode, keys):
    with open(path, mode) as f:
        for k in keys:
            line = k + ',' + ','.join([str(p) for p in items[k]]) + '\n' 
            f.write(line)
          

#print both term and its number of occurence
def print_vocab(d, path, mode, keys):
    with open(path, mode) as f:
        for k in keys:
            line = k + ',' + str(d[k][0]) + '\n'
            f.write(line)
            


#print pos (posting or position) from an item 
#item can be list, set or dictionary
def print_posting(d, path, mode, keys, word_type):    
    if word_type == 'term':
        with open(path, mode) as f:
            for k in keys:
                line = k +  ',' + ','.join([str(p) for p in d[k][1]]) + '\n' 
                f.write(line)
    
    elif word_type == 'sw':
        with open(path, mode) as f:
            for k in keys:
                line = k +  ',' + ','.join([str(p) for p in d[k]]) + '\n' 
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
    
    #remove decimal number
    #but ignore stuff like IP adress
    #got it from stackoverflown
    #god damn regex is hard
    s = re.sub(r'(?<!\S)\d+\.\d+\b(?!\.\d)', '',s)
    
    
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
    tokens = []
    
    #PARSING LINE BY LINE
    #record the ending position
    #then print out endpos, whole line

    L_line = []
    
    with open(doc_path, 'r') as f:
        p = 0
        for line in f:
            if not line: 
                continue
            line_tk = line_tokenise(line)         
            change = False
            #count position in each line
            for i, tk in enumerate(line_tk):
                tokens.append(tk)
                if tk in not_index:
                    continue
                elif tk == "'s" and i > 0 and line_tk[i-1] not in ('he', 'she', 'it'):
                    continue
                p += 1
                change = True
            
            #get line and end position
            if change: 
                L_line.append(str(p)+ ',' + line)   
            
                
    with open(os.path.join(doc_line_dir, str(docID)), 'w') as f:
        f.writelines(L_line)
    
    
      
    #pos_tag, lemmatise, and index
    tokens_tag = nltk.pos_tag(tokens)
    position = 0
    
    for i, tk in enumerate(tokens_tag):
        term = index_term(tk[0], tk[1])
        if term: 
            position += 1
            if term in stopwords:
                add_position(sw_pos, term, position)
                add_posting(posting_sw, term, docID, word_type='sw')
            else:
                add_position(term_pos, term, position)
                add_posting(posting_term, term, docID, word_type='term')
            
    sorted_terms = sorted(term_pos.keys())
    

    #print it out
    print_position(term_pos,  
                   path = os.path.join(doc_term_position_dir, str(docID)), 
                   mode = 'w',
                   keys = sorted_terms)
    
    print_position(sw_pos,
                   path = os.path.join(doc_sw_position_dir, str(docID)),
                   mode = 'w',
                   keys = [sw for sw in stopwords if sw in sw_pos.keys()])
       
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
    firstchar = '0123456789abcdefghijklmnopqrstuvwxyz'
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
                    print_posting(posting_term, 
                                  path = os.path.join(term_posting_dir, firstchar[i]), 
                                  mode = 'w', 
                                  keys = curblock,
                                  word_type='term')    
                    print_vocab(posting_term, 
                                path = os.path.join(vocab_dir, firstchar[i]), 
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
        print_posting(posting_term, 
                      path = os.path.join(term_posting_dir, firstchar[i]), 
                      mode = 'w', 
                      keys = curblock,
                      word_type='term')
        print_vocab(posting_term, 
                    path = os.path.join(vocab_dir, firstchar[i]), 
                    mode = 'w', 
                    keys = curblock)



#___________MAIN BODY______________________

#___1___GETTING PROVIDED PATH FOR DATA AND INDEX

doc_dir = sys.argv[1]
index_folder = sys.argv[2]






#___2___SETTING UP PATHS FOR INDEX FOLDER AND SUBFOLDERS 
#FOR THIS PROGRAM: MAIN INDEX FOLDER ONLY HAVE 1 FILE (contain path to raw data)
#AND OTHER SUBFOLDERS
#THESE SUBFOLDERS CANNOT CONTAIN ANY SUBFOLDERS, only files with integer name

index_dir = create_dir(parent = False, child = index_folder)

#to store the specific position of a term/stopword in each doc
doc_term_position_dir = create_dir(index_dir, "doc_term_position")
doc_sw_position_dir = create_dir(index_dir, "doc_sw_position")


#POSTING FOR MAIN TERMS AND STOPWORDS
term_posting_dir = create_dir(index_dir, "term_doc")
sw_posting_dir = create_dir(index_dir, "sw_doc")

#FOlDER FOR FUCKING LINES
doc_line_dir = create_dir(index_dir, 'doc_line_endpos')

#FOLDER FOR VOCAB
vocab_dir = create_dir(index_dir, 'vocabulary')      
         


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
print_posting(posting_sw, 
              path = os.path.join(sw_posting_dir, '1'), 
              mode = 'w', 
              keys = [sw for sw in stopwords if sw in posting_sw.keys()], 
              word_type = 'sw')


#get the total frequency of all terms:
sumf = 0
for term in posting_term:
    if term[0] in 'abcdefghijklmnopqrstuvwxyz':
        sumf += posting_term[term][0]
        


#write the path of doc 
#and total frequency of all terms 
#together in one documents 
#named reference
with open(os.path.join(index_dir, "reference.txt"), 'w') as f:
          f.write(doc_dir + '\n')
          f.write(str(sumf) + '\n')
          
 


#check time            
#print("running time to this point is %s seconds" % (time.time() - start_time))        
        
 
      


      