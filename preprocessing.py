'''
script to preprocess given text file/article/comments
'''
import re
import torch
import requests.__version__
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.utils.np_utils import to_categorical
from torch.utils.data import Dataset


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def get_train_test_split(content_path, comments_path, validation_split = 0.25):
    '''
    Function used for preprocessing of given data.

    Input: 
        content_path (str) : path for content csv.
        comment_path (str) : path for comment csv.
        validation split (float) : split ratio of test and train (default: 0.25)
    '''

    #read content and comment csv
    data_train = pd.read_csv(content_path, sep="\t")
    comments_train = pd.read_csv(comments_path, sep="\t")

    contents = []
    labels = []
    texts = []
    ids = []

    for idx in range(data_train.content.shape[0]):
        text = BeautifulSoup(data_train.content[idx], features="html5lib")
        text = clean_str(text.get_text())
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        contents.append(sentences)
        ids.append(data_train.id[idx])

        labels.append(data_train.label[idx])

    labels = np.asarray(labels)
    labels = to_categorical(labels)

    # load user comments
    comments = []
    comments_text = []
    content_ids = set(ids)

    for idx in range(comments_train.comment.shape[0]):
        if comments_train.id[idx] in  content_ids:
            com_text = BeautifulSoup(comments_train.comment[idx], features="html5lib")
            com_text = clean_str(com_text.get_text())
            tmp_comments = []
            for ct in com_text.split('::'):
                tmp_comments.append(ct)
            comments.append(tmp_comments)
            comments_text.extend(tmp_comments)

    id_train, id_val, x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(ids,contents, labels, comments,
                                                                    test_size=validation_split, random_state=42)

    return {'train': {'id': id_train, 'x':x_train, 'c': c_train, 'y': y_train}, 'val': {'id': id_val, 'x':x_val, 'c': c_val, 'y': y_val}}


class FakeNewsDataset(Dataset):
    def __init__(self, content, comment, label):
        '''
        This is a custom dataset class required.
        '''
        super(FakeNewsDataset, self).__init__()
        self.content = content
        self.comment = comment
        self.label = label

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        #get content and comment at given index.
        x = self.content[index]
        c = self.comment[index]
        y = self.label[index]

        x = torch.from_numpy(x).type(torch.LongTensor)
        c = torch.from_numpy(c).type(torch.LongTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        _, y = torch.max(y, 0)

        sample = {'content': x, 'comment': c, 'label': y }

        return sample

