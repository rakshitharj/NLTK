import nltk
#nltk.download()
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import *
from nltk import *
f=open('earth.txt','r',errors = 'ignore')
text_1=f.read()

#sentence tokenizer and work tokenizer
print(sent_tokenize(text_1))
words=(word_tokenize(text_1))

#pos tagging for each word
token_tag= pos_tag(words)
print(token_tag)

#chunking: categorize different tokens into same group
grammar= "NP: {<DT>?<JJ>*<NN>}"
cp=nltk.RegexpParser(grammar)
result= cp.parse(token_tag)
print(result)
result.draw()

#stemming: normalization for words eg: eat-eating, eats,eaten
ps=PorterStemmer()
for w in words:
    rootWord=ps.stem(w)
    print(rootWord)

#Lemmatozation is algorithm way to find the lemma of word
#its better than Stemming(suffix of the word is cut)
ls=WordNetLemmatizer()
for w in words:
    rootWord=ls.lemmatize(w)
    print(rootWord)