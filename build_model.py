'''
script to build dEFEND
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_embeddeding_layer(weights_matrix, non_trainable=False):
    '''
        function to create a embedding layer from given weight matrix.

        Inputs: 
            weight_matrix (numpy.array) : a weight matrix, with shape = (vocab_size + 1, embedding_dim)
            non_trainable (bool):   arg to set weights for non-training (default: False) 
    '''

    #get shape of matrix
    num_embeddings, embedding_dim = weights_matrix.shape
    #convert weight_matrix numpy.array --> torch.Tensor
    #weights_matrix = torch.tensor(weights_matrix, requires_grad=True)
    weights_matrix = torch.from_numpy(weights_matrix)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    #add weights to layer
    emb_layer.load_state_dict({'weight': weights_matrix})

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        '''
            Time Distributed implementation of pytorch, 
            Here module is applied to every temporal dimension of the input
        '''
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x, lstm_weight):

        if len(x.size()) <= 2:
            return self.module(x, lstm_weight)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        
        y, lstm_weight = self.module(x_reshape, lstm_weight)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y, lstm_weight


class AttentionLayer(nn.Module):
    def __init__(self, device , input_last=200 ,attention_dim=100):
        '''
            Attention layer as propsosed in paper. 
        '''
        super(AttentionLayer, self).__init__()
        
        self.attention_dim = 100
        self.input_last = 200
        self.epsilon = torch.tensor([1e-07]).to(device)
        self.device = device
        #initialize weights
        self.W = torch.rand((self.input_last, self.attention_dim), requires_grad = False).to(device)
        self.b = torch.rand((self.attention_dim), requires_grad = False).to(device)
        self.u = torch.rand((self.attention_dim, 1), requires_grad = False).to(device)
        
    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        
        uit = torch.tanh(torch.matmul(x, self.W)+self.b)
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)
        
        # if mask is not None:
        #     # Cast the mask to floatX to avoid float64 upcasting in theano
        #     ait *= K.cast(mask, K.floatx())
        # print(ait)
        
        ait = ait/(torch.sum(ait, dim=1, keepdims=True) + self.epsilon).to(self.device)
        ait = torch.unsqueeze(ait, -1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=1)

        return output

class CoAttention(nn.Module):
    def __init__(self, device, latent_dim = 200):
        super(CoAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.k = 80
        self.Wl = torch.rand((self.latent_dim, self.latent_dim), requires_grad = False).to(device)
        
        self.Wc = torch.rand((self.k, self.latent_dim), requires_grad = False).to(device)
        self.Ws = torch.rand((self.k, self.latent_dim), requires_grad = False).to(device)
        
        self.whs = torch.rand((1, self.k), requires_grad = False).to(device)
        self.whc = torch.rand((1, self.k), requires_grad = False).to(device)
        
    def forward(self, sentence_rep, comment_rep):
        
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)
        
        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))
        L_trans = L.transpose(2, 1)

        Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        
        Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        
        As = F.softmax(torch.matmul(self.whs, Hs))
        
        Ac = F.softmax(torch.matmul(self.whc, Hc))
        
        As = As.transpose(2, 1)
        Ac = Ac.transpose(2, 1)

        co_s = torch.matmul(sentence_rep_trans, As)
        
        co_c = torch.matmul(comment_rep_trans, Ac)
        
        co_sc = torch.cat([co_s, co_c], dim=1)

        return torch.squeeze(co_sc, -1)


class SentenceEncoder(nn.Module):
    def __init__(self, embedding, device, max_sentence_length = 120, max_sentence_count = 50, batch_size = 32,embedding_dim = 100):
        '''
        Contains sentence encoder Architecture of the dEFEND.
        '''
        super(SentenceEncoder,self).__init__()

        #create a embedding layer.
        self.embedding = create_embeddeding_layer(embedding)
        self.word_lstm = nn.GRU(embedding_dim, 100, batch_first = True, bidirectional= True)
        self.attentaion = AttentionLayer(device)
        #self.word_lstm_weight = lstm_weight
        
    def forward(self, x, lstm_weight):
        x = self.embedding(x)
        x, lstm_weight = self.word_lstm(x, lstm_weight)
        print("b atten ", x.shape)
        x = self.attentaion(x)
        
        return x, lstm_weight


class CommentEncoder(nn.Module):
    def __init__(self, embedding, device,max_comment_length = 120, max_comment_count = 150, batch_size = 32,embedding_dim = 100):
        '''
        Contains comment encoder Architecture of the dEFEND.
        '''
        super(CommentEncoder,self).__init__()

        #create a embedding layer.
        self.embedding = create_embeddeding_layer(embedding)
        self.comment_lstm = nn.GRU(embedding_dim, 100, batch_first = True, bidirectional= True)
        self.attentaion = AttentionLayer(device)
        #self.comment_lstm_weight = lstm_weight
        
    def forward(self, x, lstm_weight):
        x = self.embedding(x)
        x, lstm_weight = self.comment_lstm(x, lstm_weight)
        x = self.attentaion(x)
        
        return x, lstm_weight



class ContentEncoder(nn.Module):
    def __init__(self, embedding, device ,max_sentence_length = 120, max_sentence_count = 50, batch_size = 32,embedding_dim = 100):
        '''
        Contains content encoder Architecture of the dEFEND.
        '''
        super(ContentEncoder, self).__init__()
        self.word_encoder = SentenceEncoder(embedding, device,max_sentence_length, max_sentence_count, batch_size, embedding_dim)
        self.time_distributed = TimeDistributed(self.word_encoder, True)
        self.content_lstm = nn.GRU(2*embedding_dim, 100, batch_first = True, bidirectional= True)
        #self.content_lstm_weight = lstm_weight

    def forward(self, x, content_lstm_weight, word_lstm_weight):
        print("start", x.shape)
        x, word_lstm_weight = self.time_distributed(x, word_lstm_weight)
        print('content', x.shape)
        x, content_lstm_weight = self.content_lstm(x, content_lstm_weight)

        return x, content_lstm_weight, word_lstm_weight
        

class CommentSequenceEncoder(nn.Module):
    def __init__(self, embedding,device ,max_comment_length = 120, max_comment_count = 50, batch_size = 32,embedding_dim = 100):
        '''
        Contain comment sequence encoder Architecture of the dEFEND.
        '''
        super(CommentSequenceEncoder, self).__init__()

        self.comment_encoder = CommentEncoder(embedding, device ,max_comment_length ,max_comment_count,  batch_size, embedding_dim )
        self.time_distributed = TimeDistributed(self.comment_encoder, True)

    def forward(self, x, lstm_weight):

        x, lstm_weight = self.time_distributed(x, lstm_weight)
        
        return x, lstm_weight

        
class dEFENDNet(nn.Module):
    def __init__(self, weight_matrix, max_sentence_length, max_comment_length, device, num_classes = 2, max_sentence_count = 50 ,max_comment_count = 150,batch_size = 32 ,embedding_dim = 100, latent_dim = 200):
        '''
        Contains Architecture of the dEFEND.

        torch Embedding is independent of input dims, so we can use same embedding
        matrix for both comment and article section.
        
        '''
        super(dEFENDNet,self).__init__()
        self.embedding = weight_matrix #create_embeddeding_layer(weight_matrix)
        self.content_encoder = ContentEncoder(self.embedding, device ,max_sentence_length, max_sentence_count, batch_size, embedding_dim)
        self.comment_sequence_encoder = CommentSequenceEncoder(self.embedding, device ,max_comment_length, max_comment_count, batch_size, embedding_dim)
        self.coattention = CoAttention(device, latent_dim)
        self.fc = nn.Linear(2*latent_dim, num_classes)
        self.softamx = nn.Softmax(dim = 1)

        self.word_lstm_weight = torch.zeros(2, max_sentence_count*batch_size, embedding_dim, requires_grad=False).to(device)
        self.content_lstm_weight = torch.zeros(2, batch_size, 100, requires_grad=False).to(device)
        self.comment_lstm_weight = torch.zeros(2, max_comment_count*batch_size, embedding_dim, requires_grad=False).to(device)



    def forward(self, content, comment):

        # passing through content encoder
        content, self.content_lstm_weight, self.word_lstm_weight = self.content_encoder(content, self.content_lstm_weight ,self.word_lstm_weight)

        #passing through comment lstm
        comment, self.comment_lstm_weight  = self.comment_sequence_encoder(comment, self.comment_lstm_weight)

        #pass through sentence comment co-attention.
        coatten = self.coattention(content, comment)

        preds = self.fc(coatten)

        preds = self.softamx(preds)

        return preds



if __name__ == "__main__":

    import os
    import numpy as np
    import time

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    content = torch.rand(12, 50, 120).type(torch.LongTensor).to(device)
    comment = torch.rand(12, 150, 120).type(torch.LongTensor).to(device)

    embedding_mat = np.random.randn(84574, 100)

    defend = dEFENDNet(embedding_mat, 120, 120, device, batch_size=12)

    defend = defend.to(device)
    
    since = time.time()
    pred = defend(content, comment)
    print(f"total time: {time.time() - since}")
    print(f"out shape: {pred.shape}")


    









