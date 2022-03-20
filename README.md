# DL individual project

Explaination of files:
- ``requirements.txt``: Contains all oython libraries that are required for the project.
- ``data_preprocessing.py``: Implements the preprocessing of train/test data set. It will ask user: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * for the path to source directory, user can press <em>\[Enter\]</em> if wishes for the program to use the default source directory. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * for the path to destination directory, where to save the preprocessed/clean data. User can press <em>\[Enter\]</em> if wishes for the program to use the default destination directory. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * if user wants to add padding to sentence, if yes then user will be asked what is the minimum accepted length of a sentence. In this case all sentences with length < accepted min will be added `"<pad>"` words until they have len equal to min. The default min length is 5.
- ``cnn_model.py``: It builds the cnn model as is described in the report. Then it loads data and starts training and evaluation of the model. User can select to use the cnn with bert embedding words or cnn that trains the embedding layer from scratch.
- ``rnn_model.py``:
- ``menu.py``: You can use the menu script to start training. It will ask you if you wish to pretrain your data, and make you choose between the <em>cnn</em> and <em>rnn</em> model. Finally, you can choose if you wish to use BERT word embeddings.

Each of the phase of the training/evalution phase is modular, so user if wish can run each part independent to the other by manualy calling the corresponiding script or by using the menu.

