
import data_preprocessing
import cnn_model
#import rnn_model


def main():
    preprocessing = str(input('Preprocess data?[y/n]: '))
    if preprocessing == 'y':
        data_preprocessing.prepare()
    elif preprocessing != 'n':
        print('Invalid anwser')
        return
    
    model = str(input('Select Model[cnn/rnn]: '))
    if model != 'cnn' or model != 'rnn':
        print('Invalid anwser')
        return
    
    embedding = str(input('Use bert embedding word?[y/n]: '))
    if embedding == 'y':
        embedding = True
    elif embedding == 'n':
        embedding = False
    else:
        print('Invalid anwser')
        return
    
    if model == 'cnn':
        cnn_model.start_training(embedding)
    #else:
    #    rnn_model.start_training(embedding)

if __name__ == '__main__': 
    main()