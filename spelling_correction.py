# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:31:46 2024

@author: dthtr
"""

# =============================================================================
# import string
# 
# for letter in string.ascii_lowercase:
#     # Print each letter, end with a space to display them on the same line.
#     print(letter, end="")
# 
# =============================================================================

import os 

import nltk

#my own modules
from special_vocab import stopwords
from special_vocab import irr_inflection_len3, irr_inflection, irr_plural


index_dir = 'my_index'
vocab_dir = os.path.join(index_dir, 'vocabulary')


wnlemmatise = nltk.WordNetLemmatizer().lemmatize




def loading_vocab(vocab_dir, vocab_type):
    if vocab_type == 'alphabet':
        firstchars = 'abcdefghijklmnopqrstuvwxyz' 
    
    elif vocab_type == 'digits':
        firstchars = '0123456789'

    vocab = {char:None for char in firstchars }
    
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
        


#to quickly trim the plural form of noun
#or 3rd person singular form of verb
#this is to avoid the need to use WordNet Lemmatiser, 
#which is slow AF

#this will just "trim" the plural suffix 
#and check if the trimmed form is in our index vocab


def quick_plural_trim(w, vocab):
    if len(w) <4:
        return
    
    char1 = w[0]
    
    if w.endswith('men'):
        new = w[:-3] + 'man'
        if new in vocab[char1]: return new 
    
    elif w.endswith('sses'):
        if w[:-2] in vocab[char1]: return w[:-2]
    
    elif w.endswith('ies'):
        new =  w[:-3]+'y' 
        if new in vocab[char1]: return new 
        new = w[:-1] 
        if new in vocab[char1]: return new
        
    elif w.endswith('ves'):
        new = w[:-1] 
        if new in vocab[char1]: return new
        new = w[:-3] + 'f' 
        if new in vocab[char1]: return new
        new =  w[:-3] + 'fe' 
        if new in vocab[char1]: return new
        
    elif w.endswith('ces'):
        new = w[:-3] + 'x'
        if new in vocab[char1]: return new
    
    elif w.endswith('es'):
        new = w[:-2] 
        if new in vocab[char1]: return new
        new = w[:-2] + 'is' 
        if new in vocab[char1]: return new
        
    elif w.endswith('s'):
        if w[:-1] in vocab[char1]: return w[:-1]
        
      


#assumption: anything that goes in this function
#already not exist in index vocab
#this function only check for the following case:
    # verb past tense and past participle
    # adjective superlative and comparative
    # V-ing
    # irregular plural and irregular verb inflection
#this function will produce proper English word
#but not checked with our index vocabulary
def quick_lemmatise(w):
    
    if len(w) == 3:
        if w in irr_inflection_len3:
            return irr_inflection_len3[w]
        else:
            return False
        
    #now deal with longer word
    
    if w in irr_inflection:
        return irr_inflection[w]
    
    if w in irr_plural:
        return irr_plural[w]
    
    if w.endswith('ed'):
        new = wnlemmatise(w, 'v')
        if new != w:
            return new
        else:
            return False
        
    if w.endswith('ing'):
        if w in {'dying', 'lying', 'tying'}:
            return w[0] + 'ie'
        elif len(w) < 6:
            return False
        else:
            new = wnlemmatise(w, 'v')
            if new != w:
                return new
            else:
                return False
        
    if w.endswith('er') or w.endswith('est'):
        new = wnlemmatise(w, 'v')
        if new != w:
            return new
        else:
            return False
    
    



def edit_candidate_set(original_word, max_edit, vocab, desperate = False):

    candidates = dict()
    permu = set()
    
    #steps of edits to avoid as much repetition as possible:
        #insert 
            #insert, delete, replace
        #delete
            #delete, replace
        #replace
            #replace   

    
    

    def add_to_candidates(s, edit):
        if s not in candidates or edit == 1:
            candidates[s]  =  (edit, vocab[s[0]][s])
    
            
        
    def check_edit_result(s, edit):    
        nonlocal desperate      
        
        char1 = s[0]
        if s in vocab[char1]:
            add_to_candidates(s, edit)
                 
        elif len(s) >=3 and desperate:
            new = quick_plural_trim(s, vocab)
            if new: #already exist in vocab
                add_to_candidates(new, edit)
            else:
                new = quick_lemmatise(s)
                if new and new in vocab[char1]:
                    add_to_candidates(new, edit)
                
            
            
    def insert_1char(s, edit):
        nonlocal max_edit
                
        #insert right in front of index i        
        for i in range(0, len(s)+1):
            for char in 'abcdefghijklmnopqrstuvwxyz':    
                new = s[:i] + char + s[i:]
                check_edit_result(new, edit)
                #if still at 1-distance edit
                if edit < max_edit:
                    insert_1char(new, edit + 1)
                    delete_1char(new, edit + 1, exclude = i)
                    replace_1char(new, edit + 1, exclude = i)


    def delete_1char(s, edit, exclude = False):
        #exclude is for when delete is after insertion
        #if insert right at index i (new char is at i)
        #then cannot delete at i-1, i, i+1
        #because it will overlap with 1-step or 2-step replace
        nonlocal max_edit
        nonlocal original_word
        
        if len(s) <= 1:
            return
        
        
        range_i = {i for i in range(0, len(s))}
                   
        if exclude:
            range_i = range_i - {exclude -2, exclude -1, exclude, exclude +1, exclude + 2}
            
        for i in range_i: 
            new = s[:i] + s[i+1:]
            if new != original_word:
                check_edit_result(new, edit)
                #if still at 1-distance edit
                if edit < max_edit:
                    delete_1char(new, edit + 1)
                    replace_1char(new, edit + 1)


    def replace_1char(s, edit, exclude = False):
        #exclude is for when replace is after insertion or replace
        nonlocal max_edit
        
        
        range_i = {i for i in range(0, len(s))}
        
        if exclude:
            range_i = range_i - {exclude}
        
        for i in range_i:
            for newchar in 'abcdefghijklmnopqrstuvwxyz':
                if newchar != s[i]:
                    new = s[:i] + newchar + s[i+1:]
                    if new != original_word:
                        check_edit_result(new, edit)
                        #if still at 1-distance edit
                        if edit < max_edit:
                            replace_1char(new, edit + 1, exclude = i)
    
    
    insert_1char(original_word, edit=1)
    delete_1char(original_word, edit=1)
    replace_1char(original_word, edit=1)
    
    return candidates



vocab_alphabet = loading_vocab(vocab_dir, 'alphabet')




#testing

ps = edit_candidate_set('bark', 2, vocab_alphabet, desperate= True)







