# Match-LSTM and Answer Pointer (Wang and Jiang, ICLR 2016) #
This repo attempts to reproduce the match-lstm and answer pointer experiments from the 2016 paper on the same. A lot of the preprocessing boiler code is taken from Stanford CS224D and https://github.com/MurtyShikhar/Question-Answering

Please refer to our research paper: 
[https://drive.google.com/file/d/1ro9bY9qsglspAojf7N6Kg75aQhrkkVQ2/view?usp=sharing](An RNN based approach to answer Open-Domain questions using Wikipedia)

The tensorflow model implementation is in qa_model.py

Use python 3.6.

1. To install all the dependencies from requirements.txt, run
 pip install dependecies pip -r requirements.txt

2. To preprocess the squad train and dev set, run
  python preprocess.py all
This step tokenizes the question and answer tokens with glove embedding and addional features set

3. In order to train the model, run
  python train.py [model_name]
  Here, the model_name is the name of the saved trained model.


config.py contains all the configuration details for the model

4. to extract an answer for a given factoid question, run
  python qa_answer_wiki.py [model_name]
 This file generate answers for any given question to command "Ask a question: " using the model_name which is already a trained model that was saved
