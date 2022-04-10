import torch
from torchtext.legacy import data
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
import functions

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()      
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                              for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text) #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1)) #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

class CNN_bert(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, 
                 dropout, bert):
        
        super().__init__()    

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                              for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1)) #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

def start_training(embeddings=False, data_path='../data/clean_archive'):
    if embeddings:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if embeddings:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id

        ID = data.LabelField(dtype = torch.float) # ignore
        TEXT = data.Field(batch_first = True,
                        use_vocab = False,
                        tokenize = function.tokenize_and_cut,
                        preprocessing = tokenizer.convert_tokens_to_ids,
                        init_token = init_token_idx,
                        eos_token = eos_token_idx,
                        pad_token = pad_token_idx,
                        unk_token = unk_token_idx)
    else:
        ID = data.LabelField(dtype = torch.float) # ignore
        TEXT = data.Field(tokenize = 'spacy',
                        tokenizer_language = 'en_core_web_sm',
                        batch_first = True)


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

    if not embeddings:
        MAX_VOCAB_SIZE = 25_000 # in case process take too much, we will use less unique words

        TEXT.build_vocab(data_train,
                        vectors = "glove.6B.100d", #pre-trained 
                        unk_init = torch.Tensor.normal_)
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    
    LABEL.build_vocab(data_train)
    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                    (data_train, valid_train, data_test), 
                                    batch_size = BATCH_SIZE, 
                                    sort_key=lambda x: len(x.TEXT),
                                    shuffle=True,
                                    sort=False,
                                    sort_within_batch=True)
    
    if embeddings:
        bert = BertModel.from_pretrained('bert-base-uncased')
        n_filters = 100
        filter_sizes = [2,3,4]
        output_dim = len(LABEL.vocab)
        dropout = 0.5

        model = CNN_bert(n_filters, filter_sizes, output_dim, dropout, bert)
        print(f'The model has {functions.count_parameters(model):,} parameters in total')
        for name, param in model.named_parameters():                
            if name.startswith('bert'):
                param.requires_grad = False
        print(f'The model has {functions.count_parameters(model):,} trainable parameters')
    else:
        input_dim = len(TEXT.vocab)
        embedding_dim = 100
        n_filters = 100
        filter_sizes = [2, 3, 4]
        output_dim = len(LABEL.vocab)
        dropout = 0.5
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)

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

    n_epochs = 30
    best_valid_loss = float('inf')
    #history_loss = []
    #history_acc = []
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = functions.train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = functions.evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = functions.epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut5-model.pt')
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        #history_loss.append(f'{valid_loss:.3f}')
        #history_acc.append(f'{valid_acc*100:.2f}')
    
    model.load_state_dict(torch.load('tut5-model.pt'))
    test_loss, test_acc = functions.evaluate(model, test_iterator, criterion, True)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    #print(history_loss)
    #print(history_acc)

def main():
    start_training(False)

if __name__ == '__main__': 
    main()