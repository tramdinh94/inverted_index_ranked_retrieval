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



wnlemmatise = nltk.WordNetLemmatizer().lemmatize

#test data: sumf = 157796 word occurence



        


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
    
    


#inputs: query_word, maximum distance edit, vocab
#output: a dictionary 
    #key: candidate word (exists in vocab)
    #value: tuple (min_edit_distance, frequency)
def correct_candidate_set(query_word, max_edit, vocab, desperate):

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
        nonlocal query_word
        
        if len(s) <= 1:
            return
        
        
        range_i = {i for i in range(0, len(s))}
                   
        if exclude:
            range_i = range_i - {exclude -2, exclude -1, exclude, exclude +1, exclude + 2}
            
        for i in range_i: 
            new = s[:i] + s[i+1:]
            if new != query_word:
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
                    if new != query_word:
                        check_edit_result(new, edit)
                        #if still at 1-distance edit
                        if edit < max_edit:
                            replace_1char(new, edit + 1, exclude = i)
    
    
    insert_1char(query_word, edit=1)
    delete_1char(query_word, edit=1)
    replace_1char(query_word, edit=1)
    
    return candidates




######______NOW THE MAIN PART____________________________________________
#####_______NEED TO OUTPUT LIST OF SPELLING CORRECTION CANDIDATES_________
####________RANKEDDDDD___________________________________________________
#produce a rank number proportional to probability that a word is correct
#given data (actual error)
#a very very very rudimental error model
#because I am so dead don't wanna code anymore

#very LAZY ASSUMPTIONs:
    # P(word) = frequency in vocab/all word frequency in vocab 
        # this set of data has in total 157796 term occurences
        # may be should just calculate this when indexing
        # cause indexing has 60 seconds to run
    # given a correct word, probability of error is P(error)
        # which, honestly in this case depends on the professor
        # since 26 marks is for query with no error, 10 with obvious error, 4 for not obvious error
        # and not all words in query with errors would have error
        # I assume:
            # P(correct query) = 0.65 ~ 26/40            
            # P(obviously wrong query) = 0.25  ~ 10/40 
            # P(not obviously wrong query ) = 0.1 = 4/10
            
          
    #given any error (obvious or not), there is 80% chance an error within 1-edit distance, 
          #and 20% chance in 2 edit distance
    
    #now, I am lazy, so won't calculate the possibility of error at character level
    
    #so: P(query|correct) takes the following form
        # if query == original (means edit = 0)
            # P(query|original) = 0.65  (really, can take it as probability of edit = 0)
            
        # if query is obviously wrong (not in vocab)
            # P(obviously_wrong_query|original) = 0.25*P(edit-distance)
            
        # if query is not obviously wrong (in vocab)    
            # P(not_obviously_wrong_query|original) = 0.1*P(edit-distance)

      
    #so: the non-normalise posterior probability is:
        # P(original|query) ~ P(query|original) * P(original)

#anyways, bayesian interpretation is that probability reflect our belief about stuff
#I don't know much, so this is the best I can do with my belief

def P_correct(Pw, edit, obvious):
    
    if edit == 0:   #means query = original (q = w)
        return 0.65*Pw
        
    else:
        P_error = 0.25 if obvious else 0.1
        P_edit = 0.8 if edit == 1 else 0.2
        return Pw*P_error*P_edit
    
    

def main(q, vocab, sumf, max_edit = 2):
    if q in vocab[q[0]]:
        obvious = False
    else:
        obvious = True
    
    if obvious: 
        desperate = True
    else:
        desperate = False
        
    candidates = correct_candidate_set(q, max_edit, vocab, desperate)
    L = []
    for w in candidates:
        edit = candidates[w][0]
        Pw = candidates[w][1]/sumf
        L.append((w, P_correct(Pw, edit, obvious)))
        
    L.sort(key = lambda x: x[1], reverse=True)
    
    if obvious:
        return L[0]
    else:
        Pq = vocab[q[0]][q]/sumf
        P_no_error = P_correct(Pq, 0, obvious)  #0 edit
        result = [x for x in L if x[1] > P_no_error]
        if result:
            return P_no_error, result
        else:
            return False
            
    
