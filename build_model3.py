'''
script to build dEFEND
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    else:
        emb_layer.weight.requires_grad = True

    return emb_layer


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
        #initialize parametres
        self.W = nn.Parameter(torch.Tensor((input_last, attention_dim)))
        self.b = nn.Parameter(torch.Tensor((attention_dim)))
        self.u = nn.Parameter(torch.Tensor((attention_dim, 1)))

        #register params
        self.register_parameter("W", self.W)
        self.register_parameter("b", self.b)
        self.register_parameter("u", self.u)

        #initialize param data
        self.W.data = torch.randn((input_last, attention_dim))
        self.b.data = torch.randn((attention_dim))
        self.u.data = torch.randn((attention_dim, 1))
                
    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        
        uit = torch.tanh(torch.matmul(x, self.W)+ self.b)
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
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))

        #register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)


        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        
    def forward(self, sentence_rep, comment_rep):
        
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)
        
        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))
        L_trans = L.transpose(2, 1)

        Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        
        Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        
        As = F.softmax(torch.matmul(self.whs, Hs), dim = 2)
        
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        
        As = As.transpose(2, 1)
        Ac = Ac.transpose(2, 1)

        co_s = torch.matmul(sentence_rep_trans, As)
        
        co_c = torch.matmul(comment_rep_trans, Ac)
        
        co_sc = torch.cat([co_s, co_c], dim=1)

        return torch.squeeze(co_sc, -1)


class dEFENDNet(nn.Module):
    def __init__(self, weight_matrix, max_sentence_length, max_comment_length, device, num_classes = 2, max_sentence_count = 50 ,max_comment_count = 150,batch_size = 32 ,embedding_dim = 100, latent_dim = 200):
        '''
        Contains Architecture of the dEFEND.

        torch Embedding is independent of input dims, so we can use same embedding
        matrix for both comment and article section.
        
        '''
        super(dEFENDNet,self).__init__()

        self.max_sentence_length = max_sentence_length
        self.max_comment_length = max_comment_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.device = device

        self.embedding_content = create_embeddeding_layer(weight_matrix)
        self.embedding_comment = create_embeddeding_layer(weight_matrix)

        self.sentence_encoder = nn.GRU(embedding_dim, 100, batch_first=True, bidirectional=True)
        self.comment_encoder = nn.GRU(embedding_dim, 100, batch_first=True, bidirectional=True)
        self.content_encoder = nn.GRU(input_size=2*embedding_dim, hidden_size=100, batch_first = True, bidirectional= True)
        self.attention = AttentionLayer(device)

        self.coattention = CoAttention(device, latent_dim)
        self.fc = nn.Linear(2*latent_dim, num_classes)
        self.softamx = nn.Softmax(dim = 1)

        self.attention_dim = 100
        self.input_last = 200
        
        #coattention weight
        self.latent_dim = latent_dim
        self.k = 80
        self.Wl = Variable(torch.rand((self.latent_dim, self.latent_dim), requires_grad = True).to(device))
        
        self.Wc = Variable(torch.rand((self.k, self.latent_dim), requires_grad = True).to(device))
        self.Ws = Variable(torch.rand((self.k, self.latent_dim), requires_grad = True).to(device))
        
        self.whs = Variable(torch.rand((1, self.k), requires_grad = True).to(device))
        self.whc = Variable(torch.rand((1, self.k), requires_grad = True).to(device))
        


    def forward(self, content, comment, weight):

        embedded_content = self.embedding_content(content)
        embedded_comment = self.embedding_comment(comment)

        #print("embedded content weights:", self.embedding_content.weight[0][0])
        #print("embedded comment weights:", self.embedding_comment.weight[0][0])
           
        embedded_comment = embedded_comment.view(-1, self.max_sentence_length, self.embedding_dim)
        embedded_content = embedded_content.view(-1, self.max_comment_length, self.embedding_dim)

        x1, word_lstm_weight = self.sentence_encoder(embedded_content)
        #print("word lstm weights:", word_lstm_weight[0][0][0])
        xa = self.attention(x1)
        
        print("comment lstm weights", self.comment_encoder.hidden[0][0][0])
        x2, comment_lstm_weight = self.comment_encoder(embedded_comment)
        xc = self.attention(x2)
        #print("comment lstm weights:", comment_lstm_weight[0][0][0])

        xa = xa.view(-1, self.max_sentence_count, 2*self.embedding_dim)
        xc = xc.view(-1, self.max_comment_count, 2*self.embedding_dim)
        
        x3, content_lstm_weight = self.content_encoder(xa)
        
        
        #print("content lstm weights:", content_lstm_weight[0][0][0])
        #pass through sentence comment co-attention.
        coatten = self.coattention(x3, xa)

        preds = self.fc(coatten)

        preds = self.softamx(preds)

        return preds, (word_lstm_weight, comment_lstm_weight, content_lstm_weight)

    def initHidden(self):
        
        word_lstm_weight = Variable(torch.zeros(2, self.max_sentence_count*self.batch_size, self.embedding_dim).to(self.device))
        comment_lstm_weight = Variable(torch.zeros(2, self.max_comment_count*self.batch_size, self.embedding_dim).to(self.device))
        content_lstm_weight = Variable(torch.zeros(2, self.batch_size, self.embedding_dim).to(self.device))

        return (word_lstm_weight, comment_lstm_weight, content_lstm_weight)
    
    



if __name__ == "__main__":

    import os
    import numpy as np
    import time

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    content = torch.rand(12, 50, 120).type(torch.LongTensor).to(device)
    comment = torch.rand(12, 150, 120).type(torch.LongTensor).to(device)

    embedding_mat = np.random.randn(84574, 100)

    defend = dEFENDNet(embedding_mat, 120, 120, device, batch_size=12)

    for name, param in defend.named_parameters():
        print(name)

    defend = defend.to(device)
    
    since = time.time()
    pred = defend(content, comment)
    print(f"total time: {time.time() - since}")
    print(f"out shape: {pred.shape}")


    









