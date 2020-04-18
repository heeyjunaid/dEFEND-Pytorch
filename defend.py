'''
file contains all required function for defend to train.
'''
import os
import torch
import time
import copy
import math
import gc
from torch import nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import keras_preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from build_model2 import dEFENDNet
from preprocessing import FakeNewsDataset

class Metrics():
    def __init__(self, platform):

        pass
        
    def on_train_begin(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []

    def on_batch_end(self,epoch, batch ,val_predict, val_targ):

        # val_predict_onehot = (
        #     np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        # val_targ_onehot = self.validation_data[2]

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        #_val_auc = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #self.val_auc.append(_val_auc)
        self.val_acc.append(_val_acc)

        print("Epoch: %d - batch: %d- val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f " % (
            epoch, batch ,_val_acc, _val_precision, _val_recall, _val_f1))


    def on_epoch_end(self, epoch):

        f1 = np.mean(self.val_f1s)
        recall = np.mean(self.val_recalls)
        precision = np.mean(self.val_precisions)
        acc = np.mean(self.val_acc)
        
        print('-' * 100)
        print("Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f " % (
            epoch, acc, precision, recall, f1))
        print('-' * 100)
        return acc


class Defend():
    def __init__(self, platform, MAX_SENTENCE_LENGTH, MAX_COMS_LENGTH):
        self.model = None
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.MAX_SENTENCE_COUNT = 50
        self.MAX_COMS_COUNT = 150
        self.MAX_COMS_LENGTH = MAX_COMS_LENGTH
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.sentence_comment_co_model = None
        self.tokenizer = None
        self.class_count = 2
        self.metrics = Metrics(platform)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        

    def _fit_on_texts_and_comments(self, train_x, train_c, val_x, val_c):
        """
        Creates vocabulary set from the news content and the comments
        """
        texts = []
        texts.extend(train_x)
        texts.extend(val_x)
        comments = []
        comments.extend(train_c)
        comments.extend(val_c)
        self.tokenizer = keras_preprocessing.text.Tokenizer(num_words=20000)
        all_text = []

        all_sentences = []
        for text in texts:
            for sentence in text:
                all_sentences.append(sentence)

        all_comments = []
        for com in comments:
            for sentence in com:
                all_comments.append(sentence)

        all_text.extend(all_comments)
        all_text.extend(all_sentences)
        self.tokenizer.fit_on_texts(all_text)
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()


    def _create_reverse_word_index(self):
        '''
            create a dictionary with index as key and corresponding word as value pair.
            e.g.

            reverse_word_index = {1: 'the', 2: 'to', 3: 'a', 4: 'and', 5: 'of', 6: 'is', 7: 'in', 8: 'that', 9: 'i', ....}
        '''
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}


    def _build_model(self, n_classes=2, batch_size = 12,embedding_dim=100, embeddings_path="./", aff_dim=80):
        '''
            This function is used to build dEFEND model.
        '''
        embeddings_index = {}

        #open embedding file and read data
        f = open(os.path.join(embeddings_path, 'glove.6B.100d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        #get word index
        word_index = self.tokenizer.word_index
        embedding_matrix = np.random.random((len(word_index)+1, embedding_dim))

        #create embedding matrix.
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        model = dEFENDNet(embedding_matrix, self.MAX_SENTENCE_LENGTH, self.MAX_COMS_LENGTH, self.device, batch_size = batch_size)

        for name, param in model.named_parameters():
            print(name)


        model = model.to(self.device)

        self.optimizer = optim.SGD(model.parameters(), lr = 0.1)
        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.01)
        self.criterion = CrossEntropyLoss()

        return model

    def _encode_texts(self, texts):
        """
        Pre process the news content sentences to equal length for feeding to GRU
        :param texts:
        :return:
        """
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32')
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_SENTENCE_LENGTH, padding='post', truncating='post', value=0))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def _encode_comments(self, comments):
        """
        Pre process the comments to equal length for feeding to GRU
        """
        encoded_texts = np.zeros((len(comments), self.MAX_COMS_COUNT, self.MAX_COMS_LENGTH), dtype='int32')
        for i, text in enumerate(comments):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_COMS_LENGTH, padding='post', truncating='post', value=0))[:self.MAX_COMS_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts


    def train(self, train_x, train_y, train_c, val_c, val_x, val_y,
              batch_size=9, epochs=5,
              embeddings_path=False,
              saved_model_dir='./saved_models/', saved_model_filename="./dEFEND_saved.pt", ):

        # Fit the vocabulary set on the content and comments
        self._fit_on_texts_and_comments(train_x, train_c, val_x, val_c)

        print("building model....")
        self.model = self._build_model(n_classes=train_y.shape[-1], batch_size= batch_size, embedding_dim=100)
        print("Done.")
        print(self.model)

        print("Encoding texts....")
        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        encoded_train_c = self._encode_comments(train_c)
        encoded_val_c = self._encode_comments(val_c)

        print("preparing dataset...")
        train_dataset = FakeNewsDataset(encoded_train_x, encoded_train_c, train_y)
        val_dataset = FakeNewsDataset(encoded_val_x, encoded_val_c, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.dataset_sizes = {'train': train_dataset.__len__(), 'val': val_dataset.__len__()}
        self.dataloaders = {'train': train_loader, 'val': val_loader}
        print("Done.")

        #train model for given epoch
        self.run_epoch(epochs)

        #save model
        save_model = self.model.to('cpu')
        save_path = os.path.join(saved_model_dir, saved_model_filename)
        torch.save(save_model, save_path)
        print(f"model save at {save_path}")


    def run_epoch(self, epochs):
        '''
        Function to train model for given epochs
        '''
        #TODO: Add weight decay and step.
        #TODO: complete training script.

        since = time.time()
        clip = 5
        
        hidden = self.model.initHidden()

        self.model.sentence_encoder.hidden = hidden[0]
        self.model.comment_encoder.hidden = hidden[1]
        self.model.content_encoder.hidden = hidden[2]

        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 100)
            self.metrics.on_train_begin()

            self.model.train()
            
            for sample in self.dataloaders['train']:
                #self.model.zero_grad()
                
                comment = sample['comment'].to(self.device)
                content = sample['content'].to(self.device)
                label = sample['label'].to(self.device)
                
                output, _ = self.model(content, comment)

                loss = self.criterion(output, label)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()

            with torch.no_grad():

                self.model.eval()
                for i, sample in enumerate(self.dataloaders['val']):
                    comment = sample['comment'].to(self.device)
                    content = sample['content'].to(self.device)
                    label = sample['label'].to(self.device)
                    
                    output, _ = self.model(content, comment)

                    _, output = torch.max(output, 1)

                    output = output.detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
                    self.metrics.on_batch_end(epoch, i, output, label)

                acc_ = self.metrics.on_epoch_end(epoch) 
                if acc_ > best_acc:
                    print(f"Best Accuracy: {acc_}")
                    best_model = self.model
                    best_acc = acc_

        print("Training  end")
        print('-'*100)    
        self.model = best_model


        
    def process_atten_weight(self, encoded_text, content_word_level_attentions, sentence_co_attention):
        '''
            Process attention weights for sentence
        '''
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append((wd, content_word_level_attentions[k][i][j]))

                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        # Normalize without padding tokens
        no_pad_text_att_normalize = None
        for npta in no_pad_text_att:
            if len(npta) == 0:
                continue
            sen_att, sen_weight = list(zip(*npta))
            new_sen_weight = [float(i) / sum(sen_weight) for i in sen_weight]
            new_sen_att = []
            for sw in sen_att:
                word_list, att_list = list(zip(*sw))
                att_list = [float(i) / sum(att_list) for i in att_list]
                new_wd_att = list(zip(word_list, att_list))
                new_sen_att.append(new_wd_att)
            no_pad_text_att_normalize = list(zip(new_sen_att, new_sen_weight))

        return no_pad_text_att_normalize

    def process_atten_weight_com(self, encoded_text, sentence_co_attention):
        '''
            Process attention weight for comments
        '''
        
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append(wd)
                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        return no_pad_text_att

    # def activation_maps(self, news_article_sentence_list, news_article_comment_list):
    #     '''
    #     To get activation maps weights.
    #     '''
    #     encoded_text = self._encode_texts(news_article_sentence_list)
    #     encoded_comment = self._encode_comments(news_article_comment_list)
    #     content_word_level_attentions = []





          



              



def memReport():
    print(f"Total memory allocated: {torch.cuda.memory_allocated()/1000000000}")
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())