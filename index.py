# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:07:54 2024

@author: dthtr
"""

import sys
import os 
from collections import namedtuple
from nltk import word_tokenize
from nltk import TreebankWordTokenizer
from nltk import PunktWordTokenizer
from nktk import WordPunctTokenizer
from nltk import RegexpTokenizer



#creating a named tuple with 2 element
#pos - position of term in document
#line - the line corresponding to the position
Pos_Line = namedtuple('Pos_Line', ['pos', 'line'])



#creating a new dictionary subclass 
#doc ID: extra attribute
#add_pos_line: extra method to add a new pos_line to term
class doc_term_index(dict):
    def __init__(self, docID, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docID = docID
        
    #adding positional index list of a term    
    def add_pos_line(self, term, pos, line):
        pos_line = Pos_Line(pos, line)                
        if term in self:
            self[term].append(pos_line)
        else:
            self[term] = [pos_line]
    

            
            
#creating a subclass of dict for inverted index
class inverted_index(dict): 
       
    #adding positional data from a doc_term_index object
    def add_doc(self, doc):
        #for debug - remove when codes run properly
        assert isinstance(doc, doc_term_index)
        for term in doc:
            if term in self: 
                self[term][doc.ID] = doc[term]
            else:
                self[term] = {doc.docID : doc[term]}
    
    #print the inverted index to files in given directory
    def print_to_json(self, index_dir):
        pass
                
                    


def custom_tokenise(s):
    #take an unprocessed string
    #output list of token
    pass
    


def extract_data(file_name, file_path):
    doc = doc_term_index(docID = int(file_name))
    iline = 0
    itk = 0
    with open(file_path, 'r') as f:
        for line in f:
            iline += 1
            tokens = custom_tokenise(line)
            for term in tokens:
                itk += 1
                doc.add_pos_line(term, pos = itk, line = iline)
    return doc



#doc_dir = sys.argv[1]
#index_dir = sys.argv[2]
my_inverted_index = inverted_index()


doc_dir = 'data'
index_dir = 'my_index'

#build inverted index
Files = os.listdir(doc_dir)
for file_name in Files: 
    file_path = os.path.join(doc_dir, file_name)
    #creating a doc_term_index object
    doc = extract_data(file_name, file_path)
    my_inverted_index.add_doc(doc)
        

#print inverted index
my_inverted_index.print_to_json(index_dir)        



    



