# keras imports
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import Model

# std imports
import time
import gc
import os
import json

import numpy as np


class ZeroShotBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length_document, number_document_lstm, rate_drop_document_lstm, max_sequence_length_class, number_class_lstm, rate_drop_class_lstm, number_dense, rate_drop_dense, hidden_activation, validation_split_ratio, epochs, batch_size ):
        
        self.embedding_dim = embedding_dim
        
        # document
        self.max_sequence_length_document = max_sequence_length_document
        self.number_document_lstm_units = number_document_lstm
        self.rate_drop_document_lstm = rate_drop_document_lstm
        
        # class
        self.max_sequence_length_class = max_sequence_length_class
        self.number_class_lstm_units = number_class_lstm
        self.rate_drop_class_lstm = rate_drop_class_lstm
        
        # alignment layer
        self.number_dense_units = number_dense
        self.rate_drop_dense = rate_drop_dense
        self.activation_function = hidden_activation
        
        # general 
        self.validation_split_ratio = validation_split_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def save_hyperparameter(self, path):
        hyperDict =  {
                "embedding_dim" : self.embedding_dim, 
                "max_sequence_length_document" : self.max_sequence_length_document, 
                "number_document_lstm_units" : self.number_document_lstm_units, 
                "rate_drop_document_lstm" : self.rate_drop_document_lstm,
                "max_sequence_length" : self.max_sequence_length_class,
                "number_class_lstm_units" : self.number_class_lstm_units, 
                "rate_drop_class_lstm" : self.rate_drop_class_lstm, 
                "number_dense_units" : self.number_dense_units, 
                "rate_drop_dense" : self.rate_drop_dense, 
                "activation_function" : self.activation_function,
                "validation_split_ratio" : self.validation_split_ratio,
                "epochs" : self.epochs,
                "batch_size" : self.batch_size}
        with open(path, 'w') as fp:
            json.dump(hyperDict, fp)

    def preprocess(self, tokenizer, input_pair, is_related):
        """
        preprocessing
        """
        documents = [x[0].lower() for x in input_pair]
        classes = [x[1].lower() for x in input_pair]
        train_document_sequences = tokenizer.texts_to_sequences(documents)
        train_class_sequences = tokenizer.texts_to_sequences(classes)
    
        data_document = pad_sequences(train_document_sequences, maxlen=self.max_sequence_length_document)
        data_class = pad_sequences(train_class_sequences, maxlen=self.max_sequence_length_class)
        labels = np.array(is_related)
        
        return data_document, data_class, labels


    def train_model(self, input_pair, is_related, input_pair_dev, is_related_dev, embedding_matrix, tokenizer, model_save_directory='./'):
        """
        Train network to find relations between documents and classes in `input_pair`
        Args:
            input_pair (list): list of tuple of document class pairs
            is_related (list): target value 1 if pair is related otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """

        train_data_document, train_data_class, train_labels = self.preprocess(tokenizer, input_pair, is_related)
        val_data_document, val_data_class, val_labels = self.preprocess(tokenizer, input_pair_dev, is_related_dev)

        nb_words = len(tokenizer.word_index) + 1

        # Creating embedding layers
        embedding_layer_document = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_document, trainable=False)
        embedding_layer_class = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_class, trainable=False)

        # Creating LSTM Encoder layer for documents
        sequence_document_input = Input(shape=(self.max_sequence_length_document,), dtype='int32')
        embedded_sequences_document = embedding_layer_document(sequence_document_input)
        document_lstm_layer = Bidirectional(LSTM(self.number_document_lstm_units, dropout=self.rate_drop_document_lstm, recurrent_dropout=self.rate_drop_document_lstm))
        x1 = document_lstm_layer(embedded_sequences_document)

        # Creating LSTM Encoder layer for classes
        sequence_class_input = Input(shape=(self.max_sequence_length_class,), dtype='int32')
        embedded_sequences_class = embedding_layer_class(sequence_class_input)
        class_lstm_layer = Bidirectional(LSTM(self.number_class_lstm_units, dropout=self.rate_drop_class_lstm, recurrent_dropout=self.rate_drop_class_lstm))
        x2 = class_lstm_layer(embedded_sequences_class)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_document_input, sequence_class_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        STAMP = "zeroShotLabel_lstm"
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory)
        model_path = model_save_directory + STAMP + ".h5"
        model_hyperparameter_path = model_save_directory + STAMP + "_hyperparameter.json"
        self.save_hyperparameter(model_hyperparameter_path)
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False)
        
        tensorboard = TensorBoard(log_dir=model_save_directory + "logs/")

        model.fit([train_data_document, train_data_class], train_labels,
                  validation_data=([val_data_document, val_data_class], val_labels),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard], verbose=1)

        return model_path

class ZeroShotKGBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length_document, number_document_lstm, rate_drop_document_lstm, max_sequence_length_class, number_class_lstm, rate_drop_class_lstm, kg_embedding_dim, number_dense, rate_drop_dense, hidden_activation, validation_split_ratio, epochs, batch_size ):
        
        self.embedding_dim = embedding_dim
        
        # document
        self.max_sequence_length_document = max_sequence_length_document
        self.number_document_lstm_units = number_document_lstm
        self.rate_drop_document_lstm = rate_drop_document_lstm
        
        # class
        #self.max_sequence_length_class = max_sequence_length_class
        #self.number_class_lstm_units = number_class_lstm
        #self.rate_drop_class_lstm = rate_drop_class_lstm
        self.kg_embedding_dim = kg_embedding_dim

        # alignment layer
        self.number_dense_units = number_dense
        self.rate_drop_dense = rate_drop_dense
        self.activation_function = hidden_activation
        
        # general 
        self.validation_split_ratio = validation_split_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def save_hyperparameter(self, path):
        hyperDict =  {
                "embedding_dim" : self.embedding_dim, 
                "max_sequence_length_document" : self.max_sequence_length_document, 
                "number_document_lstm_units" : self.number_document_lstm_units, 
                "rate_drop_document_lstm" : self.rate_drop_document_lstm,
                #"max_sequence_length_class" : self.max_sequence_length_class,
                #"number_class_lstm_units" : self.number_class_lstm_units, 
                #"rate_drop_class_lstm" : self.rate_drop_class_lstm, 
                "kg_embedding_dim" : self.kg_embedding_dim,
                "number_dense_units" : self.number_dense_units, 
                "rate_drop_dense" : self.rate_drop_dense, 
                "activation_function" : self.activation_function,
                "validation_split_ratio" : self.validation_split_ratio,
                "epochs" : self.epochs,
                "batch_size" : self.batch_size}
        with open(path, 'w') as fp:
            json.dump(hyperDict, fp)
    
    def preprocess(self, tokenizer, input_pair, is_related):
        """
        preprocessing
        """
        documents = [x[0].lower() for x in input_pair]
        classes = [x[1][0].lower() for x in input_pair]
        classesKG = [x[1][1] for x in input_pair]

        train_document_sequences = tokenizer.texts_to_sequences(documents)
        #train_class_sequences = tokenizer.texts_to_sequences(classes)
    
        data_document = pad_sequences(train_document_sequences, maxlen=self.max_sequence_length_document)
        #data_class = pad_sequences(train_class_sequences, maxlen=self.max_sequence_length_class)
        
        classesKG = np.array(classesKG)
        labels = np.array(is_related)

        return data_document, classesKG, labels
        
    def train_model(self, input_pair, is_related, input_pair_dev, is_related_dev, embedding_matrix, kg_embedding_matrix, tokenizer, model_save_directory='./'):
        """
        Train network to find relations between documents and classes in `input_pair`
        Args:
            input_pair (list): list of tuple of document class pairs
            is_related (list): target value 1 if pair is related otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """
        
        # TODO add transformation to embedding ID here instead of main
        train_data_document, train_classKG, train_labels = self.preprocess(tokenizer, input_pair, is_related)  
        val_data_document, val_classKG, val_labels = self.preprocess(tokenizer, input_pair_dev, is_related_dev)


        nb_words = len(tokenizer.word_index) + 1
        nb_kg = 31 # TODO fix right len

        # Creating embedding layers
        embedding_layer_document = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_document, trainable=False)
        #embedding_layer_class = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_class, trainable=False)
        kg_embedding_layer_class = Embedding(nb_kg, self.kg_embedding_dim,  weights=[kg_embedding_matrix], input_length=1, trainable=False)

        # Creating LSTM Encoder layer for documents
        sequence_document_input = Input(shape=(self.max_sequence_length_document,), dtype='int32')
        embedded_sequences_document = embedding_layer_document(sequence_document_input)
        document_lstm_layer = Bidirectional(LSTM(self.number_document_lstm_units, dropout=self.rate_drop_document_lstm, recurrent_dropout=self.rate_drop_document_lstm))
        x1 = document_lstm_layer(embedded_sequences_document)

        # Creating LSTM Encoder layer for classes
        #sequence_class_input = Input(shape=(self.max_sequence_length_class,), dtype='int32')
        #embedded_sequences_class = embedding_layer_class(sequence_class_input)
        #class_lstm_layer = Bidirectional(LSTM(self.number_class_lstm_units, dropout=self.rate_drop_class_lstm, recurrent_dropout=self.rate_drop_class_lstm))
        #x2 = class_lstm_layer(embedded_sequences_class)
        # Creating KG embedding layer for classes
        class_kg_input = Input(shape=(1,),dtype='int32')
        class_kg_embedding = kg_embedding_layer_class(class_kg_input)
        x3 =  Lambda(lambda x: x[:,0,:])(class_kg_embedding)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x3])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_document_input, class_kg_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        STAMP = "zeroShotKG_lstm" 
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory)
        model_path = model_save_directory + STAMP + ".h5"
        model_hyperparameter_path = model_save_directory + STAMP + "_hyperparameter.json"
        self.save_hyperparameter(model_hyperparameter_path)
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False)
        
        tensorboard = TensorBoard(log_dir=model_save_directory + "logs/")

        model.fit([train_data_document, train_classKG], train_labels,
                  validation_data=([val_data_document, val_classKG], val_labels),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return model_path

class ZeroShotLabelKGBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length_document, number_document_lstm, rate_drop_document_lstm, max_sequence_length_class, number_class_lstm, rate_drop_class_lstm, kg_embedding_dim, number_dense, rate_drop_dense, hidden_activation, validation_split_ratio, epochs, batch_size ):
        
        self.embedding_dim = embedding_dim
        
        # document
        self.max_sequence_length_document = max_sequence_length_document
        self.number_document_lstm_units = number_document_lstm
        self.rate_drop_document_lstm = rate_drop_document_lstm
        
        # class
        self.max_sequence_length_class = max_sequence_length_class
        self.number_class_lstm_units = number_class_lstm
        self.rate_drop_class_lstm = rate_drop_class_lstm
        self.kg_embedding_dim = kg_embedding_dim

        # alignment layer
        self.number_dense_units = number_dense
        self.rate_drop_dense = rate_drop_dense
        self.activation_function = hidden_activation
        
        # general 
        self.validation_split_ratio = validation_split_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def save_hyperparameter(self, path):
        hyperDict =  {
                "embedding_dim" : self.embedding_dim, 
                "max_sequence_length_document" : self.max_sequence_length_document, 
                "number_document_lstm_units" : self.number_document_lstm_units, 
                "rate_drop_document_lstm" : self.rate_drop_document_lstm,
                "max_sequence_length" : self.max_sequence_length_class,
                "number_class_lstm_units" : self.number_class_lstm_units, 
                "rate_drop_class_lstm" : self.rate_drop_class_lstm, 
                "kg_embedding_dim" : self.kg_embedding_dim,
                "number_dense_units" : self.number_dense_units, 
                "rate_drop_dense" : self.rate_drop_dense, 
                "activation_function" : self.activation_function,
                "validation_split_ratio" : self.validation_split_ratio,
                "epochs" : self.epochs,
                "batch_size" : self.batch_size}
        with open(path, 'w') as fp:
            json.dump(hyperDict, fp)
    
    def preprocess(self, tokenizer, input_pair, is_related):
        """
        preprocessing
        """
        documents = [x[0].lower() for x in input_pair]
        classes = [x[1][0].lower() for x in input_pair]
        classesKG = [x[1][1] for x in input_pair]

        train_document_sequences = tokenizer.texts_to_sequences(documents)
        train_class_sequences = tokenizer.texts_to_sequences(classes)
    
        data_document = pad_sequences(train_document_sequences, maxlen=self.max_sequence_length_document)
        data_class = pad_sequences(train_class_sequences, maxlen=self.max_sequence_length_class)
        
        classesKG = np.array(classesKG)
        labels = np.array(is_related)

        return data_document, data_class, classesKG, labels
        
    def train_model(self, input_pair, is_related, input_pair_dev, is_related_dev, embedding_matrix, kg_embedding_matrix, tokenizer, model_save_directory='./'):
        """
        Train network to find relations between documents and classes in `input_pair`
        Args:
            input_pair (list): list of tuple of document class pairs
            is_related (list): target value 1 if pair is related otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """
        
        # TODO add transformation to embedding ID here instead of main
        train_data_document, train_data_class, train_classKG, train_labels = self.preprocess(tokenizer, input_pair, is_related)  
        val_data_document, val_data_class, val_classKG, val_labels = self.preprocess(tokenizer, input_pair_dev, is_related_dev)


        nb_words = len(tokenizer.word_index) + 1
        nb_kg = 31  # TODO fix to correct len
        # Creating embedding layers
        embedding_layer_document = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_document, trainable=False)
        embedding_layer_class = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_sequence_length_class, trainable=False)
        kg_embedding_layer_class = Embedding(nb_kg, self.kg_embedding_dim,weights=[kg_embedding_matrix], input_length=1, trainable=False)

        # Creating LSTM Encoder layer for documents
        sequence_document_input = Input(shape=(self.max_sequence_length_document,), dtype='int32')
        embedded_sequences_document = embedding_layer_document(sequence_document_input)
        document_lstm_layer = Bidirectional(LSTM(self.number_document_lstm_units, dropout=self.rate_drop_document_lstm, recurrent_dropout=self.rate_drop_document_lstm))
        x1 = document_lstm_layer(embedded_sequences_document)

        # Creating LSTM Encoder layer for classes
        sequence_class_input = Input(shape=(self.max_sequence_length_class,), dtype='int32')
        embedded_sequences_class = embedding_layer_class(sequence_class_input)
        class_lstm_layer = Bidirectional(LSTM(self.number_class_lstm_units, dropout=self.rate_drop_class_lstm, recurrent_dropout=self.rate_drop_class_lstm))
        x2 = class_lstm_layer(embedded_sequences_class)
        # Creating KG embedding layer for classes
        # TODO switch to embedding_layer instead of taking embedding as direct input  
        class_kg_input = Input(shape=(1,),dtype='int32')
        class_kg_embedding = kg_embedding_layer_class(class_kg_input)
        x3 = Lambda(lambda x: x[:, 0, :])(class_kg_embedding)

        # Merging two LSTM encodes vectors from sentences to
        # pass it to dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2, x3])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_document_input, sequence_class_input, class_kg_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        STAMP = "zeroShotLabelKG_lstm"
        checkpoint_dir = model_save_directory + 'checkpoints/' 
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + STAMP + ".h5"
        model_hyperparameter_path = checkpoint_dir + STAMP + "_hyperparameter.json"
        self.save_hyperparameter(model_hyperparameter_path)
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False)
        
        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/")

        model.fit([train_data_document, train_data_class, train_classKG], train_labels,
                  validation_data=([val_data_document, val_data_class, val_classKG], val_labels),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return model_path
