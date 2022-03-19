import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import gensim
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

nlp = spacy.load('en_core_web_sm')

def depure_data(data):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)
    # Remove hastags
    data = re.sub('#\S*\s?', '', data)
    # Remove tags
    data = re.sub('@\S*\s?', '', data)
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    # covid_19 --> covid
    data = re.sub("covid_19", "covid", data)
    return data

# Remove repeating words from your dataset
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# A function to detokenize all sentences
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

# Add padding to sentences
def add_padd(sentence, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
        sentence = ""
        for i in range(len(tokenized)):
            sentence = sentence + tokenized[i]
            if i < len(tokenized) - 1:
                 sentence = sentence + " "
    return sentence           

def clean_data(data_to_process, addPadding=True, isLabel=False, min_len=5):
    temp = []
    data_to_list = data_to_process.values.tolist()
    
    for i in range(len(data_to_list)):
        new_sent = data_to_list[i]
        
        new_sent = depure_data(new_sent)
        
        if addPadding and not isLabel:
            new_sent = add_padd(new_sent, min_len)
            
        temp.append(new_sent)
        
    data_words = list(sent_to_words(temp))
    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))
    
    return data



def main():
    #source_file_clean = 'data/clean/dataset(clean).csv'
    source_dir = input('Print source directory or [Enter] to get the default')
    dest_dir = input('Print destination directory or [Enter] to get the default')
    add_padding = input('Add padding? [y/n]')
    min_len = ''
    
    if add_padding == 'y':
        min_len = int(input('Give minmum len of sentence or [Enter] to get the default'))
        add_padding = True
    else:
        add_padding = False

    if min_len == '':
        min_len = 5
    
    if source_dir == '':
        source_dir = 'data/archive'

    if dest_dir == '':
        dest_dir = 'data/clean_archive'

    source_file_train = f'{source_dir}/Corona_NLP_train.csv'
    source_file_test = f'{source_dir}/Corona_NLP_test.csv'

    #nba_clean = pd.read_csv(source_file_clean)
    nba_train = pd.read_csv(source_file_train, encoding='latin-1')
    nba_test = pd.read_csv(source_file_test, encoding='latin-1')

    #print(f'The shape of the dataset is {nba_clean.shape}.')
    print(f'The shape of the dataset is {nba_train.shape}.')
    print(f'The shape of the dataset is {nba_test.shape}.')

    clean_train = clean_data(nba_train['OriginalTweet'], addPadding=add_padding, min_len=min_len)
    clean_test = clean_data(nba_test['OriginalTweet'], addPadding=add_padding, min_len=min_len)

    clean_train_label = clean_data(nba_train['Sentiment'], isLabel=True)
    clean_test_label = clean_data(nba_test['Sentiment'], isLabel=True)

    df_train = pd.DataFrame({'Text': clean_train, 
            'Emotion': clean_train_label})

    df_test = pd.DataFrame({'Text': clean_test, 
            'Emotion': clean_test_label})

    # remove duplicates - if there are any
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()

    # remove NAN - if there are any
    df_train = df_train.dropna()
    df_train = df_train[df_train.Text != '']
    df_test = df_test.dropna()
    df_test = df_test[df_test.Text != '']

    # write new data
    df_train.to_csv(f'{dest_dir}/train.csv')
    df_test.to_csv(f'{dest_dir}/test.csv')

if __name__ == '__main__': 
    main()