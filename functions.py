import torch


'''
{'positive': 0, 'negative': 1, 'neutral': 2, 'extremely positive': 3, 'extremely negative': 4}
'''

results = {0: [0,0,0], 1: [0,0,0], 2: [0,0,0],
               3: [0,0,0], 4: [0,0,0]} #[target true, predicted true, correct true]

def categorical_accuracy(preds, y, calculate=False):
    """
    Returns accuracy per batch
    """
    top_pred = preds.argmax(1, keepdim = True)
    if calculate:
        for i in range(0,5):
            results.get(i)[0] += torch.sum(y == i).float() #target true
            predicted_classes = torch.argmax(preds, dim=1) == i
            results.get(i)[1] += torch.sum(predicted_classes).float() #predicted true
            for j in range(0, len(y)):
                if (y[j] == i) and (y[j] == top_pred[j]):
                    results.get(i)[2] += 1

    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, type='cnn'):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        
        if type == 'cnn':
            predictions = model(batch.TEXT)
        else:
            text, text_lengths = batch.TEXT
            predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.LABEL)
        acc = categorical_accuracy(predictions, batch.LABEL)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()   
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, calc=False, type='cnn'):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:

            if type == 'cnn':
                predictions = model(batch.TEXT)
            else:
                text, text_lengths = batch.TEXT
                predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.LABEL)
            acc = categorical_accuracy(predictions, batch.LABEL, calc)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    if calc:
        for i in range(0, 5):
            target_true = results.get(i)[0]
            predicted_true = results.get(i)[1]
            correct_true = results.get(i)[2]

            if target_true == 0:
                recall = 0
            else:
                recall = correct_true / target_true
            
            if predicted_true == 0:
                precission = 0
            else:
                precission = correct_true / predicted_true
            if precission + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * precission * recall / (precission + recall)
            print(f"for {i}: recall: {recall:.3f}, precission:{precission:.3f} and f1:{f1_score:.3f}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tokenize_and_cut(sentence, tokenizer, max_input_length):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens