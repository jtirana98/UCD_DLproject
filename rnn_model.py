import numpy as np
import time
import pandas as pd
import torch
from torchtext.legacy import data
import random
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import functions

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.hidden = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        text = text.T

        embedded = self.dropout(self.embedding(text))
        hidden = torch.zeros(self.n_layers * 2 ,
                            text.shape[1], self.hidden)
        cell = torch.zeros( self.n_layers * 2,
                           text.shape[1], self.hidden)
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded, (hidden, cell))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)    
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc(hidden)

class LSTM_BERT(nn.Module):
    def __init__(self, bert, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        super().__init__()
        self.bert = bert
        self.hidden = hidden_dim
        self.n_layers = n_layers
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.T
        with torch.no_grad():
            embedded = self.bert(text)[0]
        hidden = torch.zeros(self.n_layers * 2 ,
                            text.shape[1], self.hidden)
        cell = torch.zeros( self.n_layers * 2,
                           text.shape[1], self.hidden)
        packed_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.hidden = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        text = text.T
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

class GRU_BERT(nn.Module):  
    def __init__(self, bert, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        super().__init__()
        self.bert = bert
        self.hidden = hidden_dim
        self.n_layers = n_layers
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.T
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden) 

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def start_training(embeddings=False, type='lstm', data_path='../data/clean_archive'):
    undersampling = str(input('Apply Undersampling[y/n]? '))
    if undersampling == 'y':
        undersampling = True
    elif undersampling == 'n':
        undersampling = False
    else:
        print('Invalid anwser')
        return

    
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if embeddings:
        data_train, data_test, output_dim = functions.init_bert(f'{data_path}/train.csv')
    else:
        ID = data.LabelField(dtype = torch.float) # ignore
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          preprocessing = generate_bigrams,
                          batch_first=True,
                          include_lengths=True)

        LABEL = data.LabelField()


        # Load  data
        data_train = data.TabularDataset(
                        path=f'{data_path}/train.csv', format='csv',
                            skip_header = True,
                            fields=[
                                ('ID', None),
                                ('TEXT', TEXT),
                                ('LABEL', LABEL)])

        data_test = data.TabularDataset(
                        path=f'{data_path}/test.csv', format='csv',
                            skip_header = True,
                            fields=[
                                ('ID', None),
                                ('TEXT', TEXT),
                                ('LABEL', LABEL)])

    data_train, valid_train = data_train.split(random_state = random.seed(SEED))

    if undersampling:
        nba_train = pd.read_csv(f'{data_path}/train.csv', encoding='latin-1')
        labels, counts = np.unique(nba_train['Emotion'], return_counts=True)

        remove = 2500
        i = 0
        while remove > 0:
            if vars(data_train[i])['LABEL'] == labels[-1]:
                del data_train.examples[i]
                remove = remove - 1
            i = i+1
        
        remove = 1000
        print(remove)
        i = 0
        while remove > 0:
            if vars(data_train[i])['LABEL'] == labels[-3]:
                del data_train.examples[i]
                remove = remove - 1
            i = i+1
    
    if not embeddings:
        MAX_VOCAB_SIZE = 25_000 # in case process take too much, we will use less unique words

        TEXT.build_vocab(data_train,
                        vectors = "glove.6B.100d", #pre-trained 
                        unk_init = torch.Tensor.normal_)
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    
        LABEL.build_vocab(data_train)
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                    (data_train, valid_train, data_test), 
                                    batch_size = BATCH_SIZE, 
                                    sort_key=lambda x: len(x.TEXT),
                                    #repeat=False,
                                    shuffle=True,
                                    sort=False,
                                    sort_within_batch=True,
                                    device = device)
    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    n_layers = 2
    bidirectional = True
    dropout = 0.25
    if embeddings:
        bert = BertModel.from_pretrained('bert-base-uncased')
    else:
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        output_dim = len(LABEL.vocab)
    
    if type == 'lstm':
        if embeddings:
            model = model = LSTM_BERT(bert, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout)

        else:
            model = LSTM(input_dim, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout, pad_idx)
  
    else:
        if embeddings:
            model = GRU_BERT(bert, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout)
        else:
            model = GRU(input_dim, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout, pad_idx)

    if embeddings:
        for name, param in model.named_parameters():                
                if name.startswith('bert'):
                    param.requires_grad = False
    else:
        # Loading pretrained embeddings
        pretrained_embeddings = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

        # zero the initial weights of the unknown and padding tokens.
        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
        model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)


    count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {count_params} trainable parameters")

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 1
    best_valid_loss = float('inf')
    history_loss = []
    history_acc = []
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = functions.train(model, train_iterator, optimizer, criterion, type='rnn')
        valid_loss, valid_acc = functions.evaluate(model, valid_iterator, criterion, type='rnn')
        
        end_time = time.time()
        epoch_mins, epoch_secs = functions.epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'model-{type}-model.pt')
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        history_loss.append(f'{valid_loss:.3f}')
        history_acc.append(f'{valid_acc*100:.2f}')
    
    model.load_state_dict(torch.load(f'model-{type}-model.pt'))
    test_loss, test_acc = functions.evaluate(model, test_iterator, criterion, True, type='rnn')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    print(history_loss)
    print(history_acc)

def main():
    start_training(False)

if __name__ == '__main__':
    main()