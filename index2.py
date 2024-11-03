# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:10:47 2024

@author: dthtr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:07:54 2024

@author: dthtr
"""

import sys
import os 
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



#SETTING UP PATHS
#doc_dir = sys.argv[1]
#index_folder = sys.argv[2]
doc_dir = 'data'
index_folder = 'my_index'
cwd = os.getcwd()

index_dir = create_dir(parent = False, child = index_folder)

doc_line_pos_dir = create_dir(index_dir, "doc_line_endposition")

doc_term_pos_main_dir = create_dir(index_dir, "doc_term_position_main")
doc_term_pos_stopwords_dir = create_dir(index_dir, "doc_term_position_stopwords")

term_doc_main_dir = create_dir(index_dir, "term_doc_main")
term_doc_stopwords_dir = create_dir(index_dir, "term_doc_stopwords")

          
#write the path of doc
with open(os.path.join(index_dir, "raw_documents.txt"), 'w') as f:
          f.write(doc_dir + '\n')
          




#FOR TOKENISATION
#pattern capture descimal number, number with comma, and number
#tokenise_pattern = r'\d+(?:\.\d+)|\d+(?:\,\d{3})*|(?:[\w\d]+|(?:[a-z]\.){2,})\'s|(?:[a-z]\.){2,}|[\w\d]+'
pattern = r'[\w\d]+'
custom_tokeniser = RegexpTokenizer(pattern)



stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
            'so', 'than', 'too', 'very', 
            'can', 'will', 'just', 'now',
            'are', 'be', "was", "were", "is", "am"
            'could', 'did', 'does', "do", 
            'had', 'has', "have", 'may', 'might',     
            's', 't', 'd', 'll', 'm', 'o', 're', 've', 'y']

stopwords.sort()  

#a dictionary for stopword and their id 
#(which is their sorted index position)
stopword_id = {}
for i, w in enumerate((stopwords)):
    stopwords[w] = i
      

grammar_term = ["'", ".", ",","'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]
not_index = ["'", ".", ","]



#FOR LEMMATISATION
lemmatiser = nltk.WordNetLemmatizer()


#creating a named tuple with 2 element
#pos - position of term in document
#line - the line corresponding to the position
Pos_Line = namedtuple('Pos_Line', ['pos', 'line'])
Term_Doc = namedtuple('Term_Doc', ['term', 'doc'])



#data structure for a block
#created as a subclass of dictionary 
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
   
    #print doc_term_pos csv (ordered)
    #mode can either be write/w or append/a
    def print_to_txt(self, path, mode, provide_keys = False):
        if provide_keys:
            with open(path, mode) as f:
                for k in provide_keys:
                    #no need to print the key itself
                    #if the list is already provided
                    #this is in the case of stopwords
                    line = ''
                    #if the item exist in the current record
                    if k in self.keys():
                        for p in self[k]: line += ',' + str(p)
                    line += '\n'
                    f.write(line)
        else:
            with open(path, mode) as f:
                for k in sorted(self.keys()):
                    line = k if isinstance(k, str) else str(k)
                    for p in self[k]: line += ',' + str(p)
                    line += '\n'
                    f.write(line)
        
            
            
        
        
        

def postag_to_lemma(tag):
    if tag.startswith('N'):
        return 'n'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('J'):
        return "a"
    elif tag.startswith('R'):
        return "r"
    else:
        return ''
    
    

def str_preprocess(string):
    #remove number with decimals
    s = re.sub(r"\b\d+(?:\.\d+)\b", ' ', string)
    
    
    #transform all n't to not
    s = re.sub(r"(\b(?:are|could|do|does|did|has|have|is|must|should|would|was|were))n\'t\b", r' \1 not ', s)
    s = re.sub(r"\b(?:ai)n\'t\b", ' are not ', s)
    s = re.sub(r"\b(?:ca)n\'t\b", ' can not ', s)
    s = re.sub(r"\b(?:sha)n\'t\b", ' shall not ', s)
    s = re.sub(r"\b(?:wo)n\'t\b", ' will not ', s)
    
    #remove dot from abbreviation 
    for m in re.findall(r"\b((?:[a-z]\.)+)", s):
        mnew = m.replace('.', '') + ' '
        s = s.replace(m, mnew)
        

    #remove comma from large number
    for m in re.findall(r"\b\d+(?:\,\d{3})+\b", s):
        mnew = m.replace(',', '')
        s = s.replace(m, mnew)

    #first use word_tokenize cause it works best for pos-tag
    #then custom tokenize terms which are not splitted by word_tokenize
    tokens = []
    for tk in word_tokenize(s):
        if tk in grammar_term:
            tokens.append(tk)
        else:
            tokens.extend(custom_tokeniser.tokenize(tk))
    
    return tokens
            
        


def doc_parsing(doc_name, doc_path):
    
    global L_term_doc_main, L_term_doc_stopwords
    global posting_main_count
    
    #local var
    docID = int(doc_name)
    doc_main = Mydict()
    doc_stopwords = Mydict()
    itk = 0
    doctk = []
    Line_endpos = []
    sw = [0]*len(stopwords)
    
    #PARSING LINE BY LINE
    #record the position of last indexing tokens on the line
    with open(doc_path, 'r') as f:
        for line in f:
            tokens = str_preprocess(line.strip().lower())
            doctk.extend(tokens)
            #count the position in line 
            #so that we can record the position per line
            pos_count = 0
            for i in range(0, len(tokens)):
                if tokens[i] in not_index:
                    continue           
                if tokens[i] == "'s" and i>0 and tokens[i-1] not in ("he", "she", "it"):
                    continue
                pos_count +=1
                
            if pos_count > 0: 
                itk = itk + pos_count
                Line_endpos.append(str(itk) + '\n')
            else:
                Line_endpos.append('0\n')            
     
    #write line position
    lp_path = os.path.join(doc_line_pos_dir, doc_name)
    with open(lp_path, 'w') as f:
        f.writelines(Line_endpos)
            
     #LEMMATISING       
    #pos tagging
    doctk_postag = nltk.pos_tag(doctk)
    #map tagging to accepted lematise arguments (n, v, adj, adv)
    doctk_lemtag = [(tk[0], postag_to_lemma(tk[1])) for tk in doctk_postag]      
    #lemmatise the important token only
    lemmatised_doctk = []
    for tk, tag in doctk_lemtag:
        if tag == '' or tk == "'s":
            lemmatised_doctk.append(tk)
        else:
            lemmatised_doctk.append(lemmatiser.lemmatize(tk, tag))
            
    pos = 0
    for i in range(0, len(lemmatised_doctk)):
        tk = lemmatised_doctk[i]
        if tk in not_index:
            continue           
        if tk == "'s" and i>0 and lemmatised_doctk[i-1] not in ("he", "she", "it"):
            continue
        if tk in ["'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]:
            tk = tk[1:]
            
        pos += 1
        #updating stopwords
        if tk in stopwords:
            L_term_doc_stopwords.add_pos(tk, docID)
            doc_stopwords.add_pos(tk, pos)

        else:
        #updating mainwords
            L_term_doc_main.add_pos(tk, docID)
            doc_main.add_pos(tk, pos)
            #proxy for block size
            posting_main_count += 1
    
    
    
    #print the doc_term_position out      
    doc_stopwords.print_to_txt(path = os.path.join(doc_term_pos_stopwords_dir, doc_name), 
                               mode = 'w')
    doc_main.print_to_txt(path = os.path.join(doc_term_pos_main_dir, doc_name), 
                          mode = 'w')
    return 





#BUILDING INVERTED INDEX USING SPIMI-Invert
#STEP 1: PARSING DOCS and write to disk
    L_term_doc_main = Mydict()
    L_term_doc_stopwords = Mydict()

    #only apply to main term
    posting_main_count = 0 
    posting_main_file = 0


    #define procedure to check and write a block when it gets big
    def block_main_write():
        nonlocal posting_main_file
        posting_main_file += 1
        f_path = os.path.join(term_doc_main_dir, str(posting_main_file))
        L_term_doc_main.print_to_txt(f_path, 'w')
        L_term_doc_main.clear()
        return
    
    def block_stopwords_write(file_num):
        f_path = os.path.join(term_doc_stopwords_dir, str(file_num))
        L_term_doc_stopwords.print_to_txt(f_path, 
                                          'w', 
                                          provide_keys=stopwords)
        L_term_doc_stopwords.clear()
        return


    #PARSING THROUGH ALL DOC
    Docs = os.listdir(doc_dir)
    #to parse doc in ordered 
    #so the docs are added in order
    Docs.sort(key = lambda x: int(x))
    for i, doc_name in enumerate(Docs): 
        doc_path = os.path.join(doc_dir, doc_name)
        #creating a doc_term_index object
        doc_parsing(doc_name, doc_path)
        if posting_main_count > 5000 or i == (len(Docs) - 1):
            block_main_write()
        if i in (99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1099, len(Docs) - 1) :
            block_stopwords_write(i//100 + 1)


   
#STEP2: MERGING FILES OF MAIN TERMS- MERGE SORT
def inverted_index_main():
    global term_doc_main_dir
    
    #create a heap and dict
    heap = []
    dterm = []
    
    
    #create a list of csv readers   
    Lreader = []
    Blocks = os.listdir(term_doc_main_dir)
    Blocks = [int(b) for b in Blocks]
    Blocks.sort()
    Lreader = [[]]*(Blocks[-1]+1)
    Lline = [[]]*(Blocks[-1]+1)
    #open file and add reader to list
    for b in Blocks:
        path = os.path.join(term_doc_main_dir, str(b))
        f = open(path, newline='\n')
        Lreader[b] = csv.reader(f, delimiter = ',')
    
    #Merge sort

    
    while Blocks:
        for b in Blocks:
            Lline[b] = Lreader[b].__next__()
            term = Lline[b][0]
            posting = Lline[b][1:]
            heapq.heappush(heap, (term, b))
        
        if heap:
            
            
        
            
            
            
            
        
    
            
        
        
        

















#MAIN BODY





