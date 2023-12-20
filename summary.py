from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer




def summarize_text(articles,dataset,titles ,num_sentences=3):
    #takes as input articles= [list of indexes of dataset] dataset, titles= [list of articles/titles]
    #output title, summary as string
    text=""
    title=""
    for index in articles:
        text=text+dataset[index] #create text with all articles and add them together
        title=title+titles[index]+". " #create title with all articles and add them together
    return luhn_summary(title,1),lex_summary(text,num_sentences) #using summary for title and text using luhn and lexrank respectivly 

def lex_summary(text,num_sentences=3):
    #takes a text as input and a num_senteces
    #and outputs a summary of the text of lenght num_sentences 
    parser = PlaintextParser.from_string(text, Tokenizer("english")) #create a parser and tokenizer  
    summarizer = LexRankSummarizer() #create lex rank summarizer
    summary = summarizer(parser.document, num_sentences) #simmarize text
    summarized_text = " ".join(str(sentence) for sentence in summary) #join text to create one single string
    return summarized_text

def luhn_summary(text,num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english")) #create a parser and tokenizer  
    summarizer = LuhnSummarizer() #create luhn summarizer 
    summary=summarizer(parser.document,num_sentences)#summarize text
    summarized_text = " ".join(str(sentence) for sentence in summary) #join text to create one single string
    return summarized_text
