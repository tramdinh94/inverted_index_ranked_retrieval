(COMP6714 assignment)
**INVERTED INDEX**
-Parse a collection of documents, preprocess the text, tokenise and build an inverted index on disk 
-In the index, words of same initial characters are grouped together in one file for faster search
-Stopwords are indexed in a separate file

**SPELLING CORRECTION**
-spelling correction is applied to query only (assignment requirement)
-includes a quick stemming function
-two types of spelling errors are dealt with: (1) obvious misspelled word and (2) non-obvious error (the misspelled string is a proper word)
-adopt some very simplified assumptions on probability of certain types of spelling error (noisy channel model)

**SEARCH PROGRAM**


