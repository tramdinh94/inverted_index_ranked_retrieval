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
from collections import namedtuple
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk import TreebankWordTokenizer
from nltk import WordPunctTokenizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

#doc_dir = sys.argv[1]
#index_dir = sys.argv[2]

#SETTING UP PATHS
doc_dir = 'data'
index_dir = 'my_index'
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

doc_line_pos_dir = os.path.join(index_dir, "doc_line_pos")
if not os.path.exists(doc_line_pos_dir):
    os.makedirs(doc_line_pos_dir)
    
doc_term_pos_dir = os.path.join(index_dir, "doc_term_pos")    
if not os.path.exists(doc_term_pos_dir):
    os.makedirs(doc_term_pos_dir)
    
term_doc_dir = os.path.join(index_dir, 'term_doc')
if not os.path.exists(term_doc_dir):
    os.makedirs(term_doc_dir)



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
            's', 't', 'd', 'll', 'm', 'o', 're', 've', 'y', 
            "'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]
 
            

grammar_term = ["'", ".", ",","'s", "'ve", "'ll", "'m", "'re", "'d", "'t"]
not_index = ["'", ".", ","]



#FOR LEMMATISATION
lemmatiser = nltk.WordNetLemmatizer()


#creating a named tuple with 2 element
#pos - position of term in document
#line - the line corresponding to the position
Pos_Line = namedtuple('Pos_Line', ['pos', 'line'])
Term_Doc = namedtuple('Term_Doc', ['term', 'doc'])





class doc_term_index(dict):
    def __init__(self, docID, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docID = docID
        
    #adding positional index list of a term    
    def add_pos(self, term, pos):
        if term in self:
            self[term].append(pos)
        else:
            self[term] = [pos]
    
    #print both term_doc and doc_term_pos csv (ordered)
    def print_to_2file(self, term_doc_file):
        sorted_term = sorted(self.keys())

        #print the doc_term_path 
        path_doctermpos = os.path.join(doc_term_pos_dir, docID)
        with open(path_doctermpos, 'w') as f:
            for term in sorted_term:
                line = term 
                for pos in self[term]: line += ',' + str(pos)
                line += '\n'
                f.write(line)
           
        #do not overwrite file, but add to existing file at the term_doc path 
        #this we have to use the name of the file provided from the caller
        path_termdoc = os.path.join(term_doc_dir, term_doc_file)
        with open(path_termdoc, "a") as myfile:
            myfile.write("appended text")
        
        
        

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
    doc_main = doc_term_index(docID = int(doc_name))
    doc_stop_word = doc_term_index(docID = int(doc_name))
    itk = 0
    doctk = []
    Line_endpos = []
    with open(doc_path, 'r') as f:
        for line in f:
            tokens = str_preprocess(line.strip().lower())
            doctk.extend(tokens)
            #count the position in line so that we can record the position per line
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
            
            
    #pos tagging
    doctk_postag = nltk.pos_tag(doctk)
    #map tagging to accepted lematise arguments (n, v, adj, adv)
    doctk_lemtag = [(tk[0], postag_to_lemma(tk[1])) for tk in doctk_postag]      
    #lemmatise the important token only
    lemmatised_doctk = []
    for tk, tag in doctk_lemtag:
        if tag != '' or tk != "'s":
            lemmatised_doctk.append(lemmatiser.lemmatize(tk, tag))
        else:
            lemmatised_doctk.append(tk)
    
    pos = 0
    for i, tk in enumerate(lemmatised_doctk):
        if tokens[i] in not_index:
            continue           
        if tokens[i] == "'s" and i>0 and tokens[i-1] not in ("he", "she", "it"):
            continue
        pos += 1
        if tk in stopwords:
            doc_stopwords.add_pos(tk, pos)
        else:
            doc_main.add_pos(tk, pos)        
    return doc_main, doc_stopwords




#build inverted index
Docs = os.listdir(doc_dir)
Docs.sort(key = lambda x: int(x))
for doc_name in Docs: 
    doc_path = os.path.join(doc_dir, doc_name)
    #creating a doc_term_index object
    doc_main, doc_stopwords = doc_parsing(doc_name, doc_path)
    
    

#print inverted index
      
