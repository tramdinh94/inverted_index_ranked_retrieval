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





def editing(w):
    pass
    





def edit_permu_set(original_word, max_edit_distance):
    global vocab
    permu_set = set()
    
    #steps of edits to avoid as much repetition as possible:
        #insert 
            #insert, delete, replace
        #delete
            #delete, replace
        #replace
            #replace   
    

    a_to_z = 'abcdefghijklmnopqrstuvwxyz'    
    set_char = 'abcdefghijklmnopqrstuvwxyz'       
            
    def insert_1char(s, edit_distance):
        nonlocal permu_set
        nonlocal max_edit_distance
        global vocab
        
        #insert right in front of index i        
        for i in range(0, len(s)+1):
            for char in 'abcdefghijklmnopqrstuvwxyz':    
                new = s[:i] + char + s[i:]
                if new in vocab:
                    permu_set.add(new)
                    #if still at 1-distance edit
                    if edit_distance < max_edit_distance:
                        insert_1char(new, edit_distance + 1)
                        delete_1char(new, edit_distance + 1, exclude = i)
                        replace_1char(new, edit_distance + 1, exclude = i)


    def delete_1char(s, edit_distance, exclude = False):
        #exclude is for when delete is after insertion
        nonlocal permu_set
        nonlocal max_edit_distance
        global vocab
        
        if len(s) <= 1:
            return
        
        if exclude:
            range_i = [i for i in range(0, len(s)) if i != exclude]
        else: 
            range_i = [i for i in range(0, len(s))]
        
        for i in range_i: 
            new = s[:i] + s[i+1:]
            if new in vocab:
                permu_set.add(new)
                #if still at 1-distance edit
                if edit_distance < max_edit_distance:
                    delete_1char(new, edit_distance + 1)
                    replace_1char(new, edit_distance + 1)


    def replace_1char(s, edit_distance, exclude = False):
        #exclude is for when replace is after insertion or replace
        nonlocal permu_set
        nonlocal max_edit_distance
        
        if exclude:
            range_i = [i for i in range(0, len(s)) if i != exclude]
        else: 
            range_i = [i for i in range(0, len(s))]
        
        for newchar in 'abcdefghijklmnopqrstuvwxyz':
            for i in range_i:
                new = s[:i] + newchar + s[i+1:]
                if new in vocab:
                    permu_set.add(new)
                    #if still at 1-distance edit
                    if edit_distance < max_edit_distance:
                        replace_1char(new, edit_distance + 1, exclude = i)
    
    
    insert_1char(original_word, edit_distance=1)
    delete_1char(original_word, edit_distance=1)
    replace_1char(original_word, edit_distance=1)
    
    return permu_set



ps = edit_permu_set('acress', max_edit_distance=2)
len(ps)


