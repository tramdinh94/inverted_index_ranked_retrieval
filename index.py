# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:07:54 2024

@author: dthtr
"""

import sys
import os 


class doc_pos_index:
    def __init__(self, docID):
        self.docID = docID
        self.pos_indexes = {}
    
    #adding positional index list of a term    
    def add(self, term, pos):
        if term in self.pos_indexes:
            self.pos_indexes[term].append(pos)
        else:
            self.pos_indexes[term] = [pos]
    
    #accessing a term's positional list
    def term_pos(self, term):
        return self.pos_indexes[term]
            
            

class inverted_index:
    def __init__(self):
        self.data = {}     #should be a dictionary of list of dictionary
    
    #adding positional data from a doc_pos_index object
    def add_doc(self, doc):    
        for term in doc.pos_indexes:
            if term in self.data: 
                #find the right position for the doc based on docID
                #and insert it
                pass
            else:
                self.data[term] = [{doc.docID: doc.term_pos(term)}]
    def print_to_json(self):
        pass
                
                    


def process():
    #take an unprocessed doc document
    #return doc_pos_index object
    pass
    


doc_dir = sys.argv[1]
index_dir = sys.argv[2]
my_inverted_index = inverted_index()




Files = os.listdir(doc_dir)
for file in Files: 
    file_path = os.path.join(doc_dir, file)
    with open(file_path, 'r') as f:
        doc_text = f.read()
        doc = process(doc_text)
        my_inverted_index.add_doc(doc)

    



