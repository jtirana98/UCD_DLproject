
import data_preprocessing
import cnn_model
#import rnn_model


def main():
    preprocessing = str(input('Preprocess data?[y/n]: '))
    if preprocessing == 'y':
        source_dir = str(input('Print source directory or [Enter] to get the default'))
        if source_dir == '':
            source_dir = 'data/archive'

        dest_dir = str(input('Print destination directory or [Enter] to get the default'))
        if dest_dir == '':
            dest_dir = 'data/clean_archive'
        data_preprocessing.prepare(source_dir, dest_dir)
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
        if dest_dir == '':
            cnn_model.start_training(embedding)
        else:
            cnn_model.start_training(embedding, dest_dir)
    #else:
    #    rnn_model.start_training(embedding)

if __name__ == '__main__': 
    main()