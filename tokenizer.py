import gensim
from gensim.models.phrases import Phrases, Phraser

bigrams=Phrases.load('models/bigrams_mov')
bibigrams=Phrases.load('models/bibigrams_mov')

def tokenizer(txt):
    tokens=txt.split(' ') 
    tokens=bibigrams[bigrams[tokens]]
    return(tokens)