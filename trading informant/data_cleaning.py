from nltk.corpus import wordnet as wn
import gensim
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

LANGUAGE = "english"
CUSTOM_STOPWORDS = ["said", "people", "year", "would", "u", "also", "say", "could", "may", "like", "told", "including", "even", "still", "go", "new","last", "take", "day"]
STOPWORDS  = nltk.corpus.stopwords.words(LANGUAGE) #+ CUSTOM_STOPWORDS

def preprocess(text, tag=False):
    lemmatizer = nltk.stem.WordNetLemmatizer() # Que anglais pour l'instant
    
    #Transformation of the text in a tokenized list of words, no maj, no non-letter char
    tokenized = gensim.utils.simple_preprocess(text)
    
    filtered = []
    if tag:
        for token, tag in nltk.pos_tag(tokenized):
            #Removing stopwords
            if token not in STOPWORDS:
                # Lemmatization based on type of word
                lemmatized = lemmatizer.lemmatize(token,penn_to_wn(tag))
    
                filtered.append(lemmatized)
    else: 
        for token in tokenized:
            #Removing stopwords
            if token not in STOPWORDS:
                # Lemmatization based on type of word
                lemmatized = lemmatizer.lemmatize(token)

                filtered.append(lemmatized)
    return " ".join(filtered)
        

def penn_to_wn(tag):
    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']


    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']


    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN
