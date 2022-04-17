# Deep Learning individual project

Install required libraries:

``pip install -r requirements.txt``

Experimental Enviroment Characteristics (I was working on):
- Python 3.18.10
- Ubuntu 20.04.4 LTS
- no CUDA 

Explaination of files:
- ``requirements.txt``: Contains all python libraries that are required for the project.
- ``menu.py``: You can use the menu script to start training. It will ask you if you wish to pretrain your data, choose model (<em>cnn</em>, <em>rnn:lstm</em> or <em>rnn:gru</em>), and finally, you can choose if you wish to use BERT word embeddings or not.
- ``data_preprocessing.py``: Implements the preprocessing of train/test data set. User is asked to give input for: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * the path to the source directory. User can press <em>\[Enter\]</em> for the default source directory ('../data/archive'). <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * the path to the destination directory, where the preprocessed/clean data are saved. User can press <em>\[Enter\]</em> for the default destination directory ('../data/clean_archive'). <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * whether user wants to add padding to a sentence. If yes then user will be asked what is the minimum accepted length of a sentence. In this case, for all sentences with length less than the accepted, `"<pad>"` words will be added until the min length is reached. The default min length is 5.
- ``cnn_model.py``: It builds the cnn model with kernels size \[2,3,4\]*100 as is described in the report. Then it loads data and starts training and evaluation of the model. User can select to use the cnn with bert embedding words or not. Note that sentences should contains at least 4 words, otherwise there will be an error during the process of convolution. So, before training with a cnn model you should add padding during the preprocessing phase.
- ``rnn_model.py``: You can choose between the GRU and the LSTM model, as they are described in the report. Also, you are asked if you wish to use BERT embedding words or not. Finally, you can choose if you want to apply undersampling on the train set. It builds the model, loads data, starts training and evaluation of the model.
- At the end of training a file with the trained weights of the final model is created. The final model, is the one with the best performance in the validation set.
- ``funcions.py``: contains functions required for the aforementioned tasks. 


All described phases are modular, thus user can run each part separately and independent to each other by manualy calling the corresponiding script or by using the menu.
