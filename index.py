# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:43:23 2024

@author: dthtr
"""

import sys
import os 
import gc

import re
import heapq
import csv


import nltk
from nltk import word_tokenize, sent_tokenize
from nltk import TreebankWordTokenizer
from nltk import WordPunctTokenizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords



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


         


###___________PREPROCESSING___________
###___________tokenisation____________
#pattern capture descimal number, number with comma, and number
#pattern = r'[\w\d]+'

pattern = r'\d+(?:\.\d+)'\
    r'|\d+(?:\,\d{3})*'\
    r'|(?:\'(?:s|ve|ll|m|re|d|t)\b)'\
    r"|[.,?!']" \
    r'|(?:[a-z]\.)+'\
    r'|(?:[\w\d]+)'\

my_tokeniser = RegexpTokenizer(pattern)


stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'almost', 'already', 'also', 'although', 'am', 'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 
             'be', 'because', 'been', 'before', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 
             'd', 'did', 'do', 'does', 'doing', 'down', 'due', 'during', 'each', 'eg', 'elsewhere', 'etc', 'even', 'ever', 'ex', 'few', 'for', 'from', 'further', 'furthermore', 
             'had', 'has', 'have', 'having', 'he', 'hence', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 
             'i', 'ie', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'may', 'me', 'meanwhile', 'might', 'more', 'moreover', 'most', 'my', 'myself', 
             'neither', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 're', 
             's', 'same', 'she', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'therefore', 'these', 'they', 
             'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'via', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 
             'will', 'with', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']         


#a dictionary for stopword and their id 
#(which is their sorted index position)
stopword_id = {stopwords[i]:i for i in range(0, len(stopwords))}
for i, w in enumerate((stopwords)):
    stopwords[w] = i
      

grammar_term = ["'", ".", ",","'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]
not_index = ["'", ".", ",", "?", "!"]



#_______lemmatisation
pos_lem_mapping = {'N': 'n', 'V': 'v', 'J':'a', 'R': 'r'}

lemmatiser = nltk.WordNetLemmatizer()


#with value of a key is a list 
class Mydict(dict):
    #adding positional index list of a term    
    # k, p = term and individual posting (docID) if dict keys = terms
    # k, P = doc and term position in a doc if dict keys = docIDs    
    
    def add_pos(self, k, p):
        if k in self:
            self[k].append(p)
        else:
            self[k] = [p]       
   
            
   

def print_pos(items, path, mode, keys = False):
    if keys:
        with open(path, mode) as f:
            for k in keys:
                if k in items:
                    line = k + ',' + ','.join([str(p) for p in items[k]]) + '\n' 
                    f.write(line)
    else:
        with open(path, mode) as f:
            for L in items:
                if L:
                    line = ','.join([str(i) for i in L]) + '\n'
                else:
                    line = '\n'
                f.write(line)
        
        
def print_term_doc(L, path):
    with open(path, 'w') as f:
        for term, docID in L:
            line = term + ',' + str(docID)+'\n'
            f.write(line)
        



def line_tokenise(line):
    #strip line, turn all into lower case, then add "EOF" to mark new line
    s = line.strip().lower() + ' EOL '
    tokens = my_tokeniser.tokenize(s)    
    return tokens


def index_term(tk, tag):
    #Check condition on the token
    #if symbol
    if tk in not_index:
        return False
       
    # if token =  's 
    # can be either verb (index) or possesive (not index)
    elif tk[0] == "'s":
        if tk[1] == 'POS': 
            return False
        else:
            term = 's'
    
    #if token has the ' symbol
    elif tk in ( "'ve", "'ll", "'m", "'re", "'d", "'t"):
        term = tk[1:]
        
    #Conditions on tag
    #if token = number
    #decimal - not index
    #other cases: remove comma (if any), and then index
    elif tag == 'CD':
        if '.' in tk: 
            return False
        else:
            term = tk.replace(',', '')  
            
    elif tag[0] in ('N', 'V', "J", 'R'):
        lw = lemmatiser.lemmatize(tk, pos_lem_mapping[tag[0]])
        term = lw
        
    else:
        term = tk
        
    return term


def doc_parsing(docID, doc_path):  
    global stopwords
    nonlocal posting_sw
    nonlocal posting_term
    
    term_pos = Mydict()
    sw_pos = Mydict()
    #PARSING LINE BY LINE
    #record the position of last indexing tokens on the line
    with open(doc_path, 'r') as f:
        tokens = [line_tokenise(line) for line in f]
    
    no_line = len(tokens)
    line_endposition = [0 for i in range(0, no_line)]  
    
    #flatten the list of tokens
    tokens = [item[i] for item in tokens for i in len(item)]
      
    #pos_tag, lemmatise, and index
    tokens_tag = nltk.pos_tag(tokens)
    position = 0
    line = 0
    
    for i, tk in enumerate(tokens_tag):
        #if it is the end of line
        if tk[0] == 'EOL':
            if i > 0 and tokens_tag[i-1] != 'EOL': 
                line_endposition[line] = position
            line += 1
            continue
        else:
            term = index_term(tk[0], tk[1])
            if term: 
                position += 1
                if term in stopwords:
                    sw_pos.add_pos(term, position)
                    sw_id = stopword_id[term]
                    posting_sw[sw_id].append(int(docID))
                else:
                    term_pos.add_pos(term, position)
                    #posting_term.add_pos(term, int(docID))
            
    sorted_term = sorted(term_pos.keys())
    
    #include it in the temp vocab
    #print it out
    print_pos(term_pos,  
              path = os.path.join(doc_term_position_dir, docID), 
              mode = 'w',
              keys = sorted_term)
    
    print_pos(sw_pos,
              path = os.path.join(doc_sw_position_dir, docID),
              mode = 'w',
              keys = stopwords)
        
    return [(t, int(docID)) for t in sorted_term]
            
     
        
     
def data_extract():
    posting_sw = []*len(stopwords)
    #posting_term = Mydict()
    term_doc = []*8
    
    #PARSING THROUGH ALL DOC
    Docs = os.listdir(doc_dir)
    #to parse doc in ordered 
    #so the docs are added in order
    Docs.sort(key = lambda x: int(x))
    for i, docID in enumerate(Docs): 
        #term_doc block
        j = i % 8
        doc_path = os.path.join(doc_dir, docID)
        sorted_term_doc = doc_parsing(docID, doc_path)               
        term_doc[j] = sorted_term_doc
        
        #after parsing every 8 documents
        #flush out blocks of data
        if j == 7:
            #flush term-doc out to disk
            term_doc = heapq.merge(term_doc[0], term_doc[1], term_doc[2], term_doc[3], 
                                   term_doc[4], term_doc[5], term_doc[6], term_doc[7])
            print_term_doc(term_doc, 
                           path = os.path.join(term_doc_dir, str(i//8)))
        
            term_doc = []*8
            
            #flush stopwords out too
            print_pos(posting_sw, 
                       keys = stopwords, 
                       path = os.path.join(sw_doc_dir, str(i//8)),
                       mode = 'w',
                       keys= False)
            
            posting_sw = []*len(stopwords)
    
    return


#function to do one round of external merge sort
#take a number of files (sorted by file name)
#create one new file or multiple new files (if final round) and delete old files
#assumption: the list of file name (Files) is sorted
def external_merge_sort_oneround(source_folder, files, final_folder = False):
                
    global index_block_size   
    
    dest_folder = final_folder if final_folder else source_folder 
    write_mode = 'w' if final_folder else 'a'
     
    #the heap will contern only the term (first value in line) 
    #and the id of the file where the line were extracted
    heap = []
    #toplines will contain the whole line in tuple form
    toplines = [ () for i in range(0, len(files))]
    Lf = []*len(files)
    
    files_path = [os.path.join(source_folder, fname) for fname in files]
    
    for i, path in enumerate(files_path):
        Lf[i] = open(path, 'r', newline='\n')
        line = Lf[i].readline().strip().split(sep = ',')
        toplines[i] = tuple(line)  #tuple so that it is immutable
        heapq.heappush(heap, (toplines[i][0], i))
    
    
    final_line = []
    cur_term = []
    while heap:
        minterm, i = heapq.heappop(heap)
        if not cur_term:
            cur_term.extend(toplines[i])
        elif minterm == cur_term[0]:
            cur_term.extend(toplines[i][1:])
        else:
            final_line.append(tuple(cur_term))
            cur_term.clear()
            cur_term.extend(toplines[i])
            
            
        #clear the popped line just to be sure
        toplines[i] = ()
        
        #push to heap the topline from file i
        try:
            next_line = Lf[i].readline()
            toplines[i] = tuple(next_line.strip().split(sep = ','))
            heapq.heappush(heap, (toplines[i][0], i))
        except EOFError:
            #if reaching the end of file 
            #then close file
            Lf[i].close()
            
        if len(final_line) >= index_block_size or len(heap) == 0:
            F_result_name = final_line[-1][0] if final_folder else 'temp'
            F_result_path = os.path.join(dest_folder, F_result_name)
    
            with open(F_result_path, mode = write_mode) as f:
                for line in final_line:
                    f.write(','.join(line) + '\n')
        
    #delete old files:
    for file in files_path:
        os.remove(file)
    
    if final_folder == False:
        os.rename(F_result_path, files_path[0])   
    
    return
            


#function to do one round of external merge WITHOUT SORTING
#this is for stopwords (already sorted)
#take a number of files (sorted by file name)
#create one new file or multile new files (if final round) and delete old files
#assumption: the list of file name (Files) is sorted
def external_merge_oneround(source_folder, files, final_folder = False):
    global stopwords
            
    dest_folder = final_folder if final_folder else source_folder 
    write_mode = 'w' if final_folder else 'a'
     
    line_block = []   
    Lf = []*len(files)
    
    files_path = [os.path.join(source_folder, fname) for fname in files]
    
    for i, path in enumerate(files_path):
        Lf[i] = open(path, 'r', newline='\n')
    
    
    for iline, sw in enumerate(stopwords):
        cur_line = ''
        for i, f in enumerate(Lf):
            s = f.readline().strip()
            cur_line += s
        line_block.append(cur_line + '\n')
        
        if len(line_block) == 64 or iline ==  len(stopwords) - 1:
            F_result_name = iline if final_folder else 'temp'
            F_result_path = os.path.join(dest_folder, F_result_name)
    
            with open(F_result_path, mode = write_mode) as f:
                f.writelines(line_block)
    
    #delete old files
    for file in files_path:
        os.remove(file)
    
    #rename the new file if needed        
    if final_folder == False:
        os.rename(F_result_path, files_path[0])   
    
    return  

    

#using the appropriate merge function to build inverted index        
def external_merging(source_folder, final_folder, max_num_run, mergeFunc):    
    Files = os.listdir(source_folder)
    Files.sort(key= lambda x: int(x))
    
    no_Files = len(Files)
    
    #intermediary merge round
    while no_Files > max_num_run:
        i = 0
        #single merge loop
        while i < no_Files:
            j = i + max_num_run  
            j = min(j, no_Files)
            if j > i+1:
                mergeFunc(source_folder, 
                          files = Files[i:j], 
                          final_folder = False)
            i = j
        
        Files = os.listdir(source_folder)
        Files.sort(key= lambda x: int(x))
        no_Files = len(Files)
    
    #final merge round
    mergeFunc(source_folder, 
              Files,
              final_folder)
    
    return





#___________MAIN BODY______________________

#___1___GETTING PROVIDED PATH FOR DATA AND INDEX
#doc_dir = sys.argv[1]
#index_folder = sys.argv[2]
doc_dir = 'data'
index_folder = 'my_index'

index_block_size = 300

cwd = os.getcwd()




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


#store temporary result
term_doc_dir = create_dir(index_dir, "term_doc")
sw_doc_dir = create_dir(index_dir, "sw_doc")

#FINAL RESULTS FOLDER
#store the inverted index (posting list only) of terms and stopwords
index_term_dir = create_dir(index_dir, "inverted_index_term")
index_sw_dir = create_dir(index_dir, "inverted_index_stopword")
        

#first, write the path of doc
with open(os.path.join(index_dir, "raw_documents.txt"), 'w') as f:
          f.write(doc_dir + '\n')
          


#__3___BUILDING INVERTED INDEX STEP BY STEP

#__3a__extract data (parsing doc one by one, flush results out to files)
data_extract()

#__3b__external merge sort to build inverted index for main terms 

#for main terms - have to sort
external_merging(source_folder = term_doc_dir,
                 final_folder = index_term_dir,
                 max_num_run = 5,
                 mergeFunc = external_merge_sort_oneround)

#for stopwords - no need to sort (because input is alreadt sorted)
external_merging(source_folder = sw_doc_dir, 
                 final_folder = index_sw_dir, 
                 max_num_run = 5, 
                 mergeFunc = external_merge_oneround)



        
            
            
       
            
        
        
        
        
        

            
            
        
        
            