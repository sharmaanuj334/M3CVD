from model import *

import numpy as np
import torch
import pandas as pd

emb_size = 224
text_embeddings = 60

from sklearn.model_selection import train_test_split

text_train, text_test, img_path_train, img_path_test, violent_train, violent_test, real_train, real_test, sentiment_train, sentiment_test = train_test_split(text, img_path, violent_label, real_label, sentiment_label, test_size=0.2, random_state=42)
text_train, text_val, img_path_train, img_path_val, violent_train, violent_val, real_train, real_val, sentiment_train, sentiment_val = train_test_split(text_train, img_path_train, violent_train, real_train, sentiment_train, test_size=0.2, random_state=42)

from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

import re

stemmer = WordNetLemmatizer()

def preprocess_text(document):
  document = re.sub(r'\W', ' ', str(document))
  document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
  document = re.sub(r'\s+', ' ', document, flags=re.I)
  document = re.sub(r'^b\s+', '', document)

  document = document.lower()

  tokens = document.split()
  tokens = [stemmer.lemmatize(word) for word in tokens]
  tokens = [word for word in tokens if word not in en_stop]
  tokens = [word for word in tokens if len(word) > 3]

  preprocessed_text = ' '.join(tokens)
  return preprocessed_text 

final_corpus = [preprocess_text(sentence) for sentence in text_train if sentence.strip() != '']

word_punctuation_tokenizer = nltk.WordPunctTokenizer()
word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

ft_model = ft_train(word_tokenized_corpus)

def embed_sentence(text):

    words = text.split()
    words = words[:60]
    word_vectors = [ft_model.wv[word] for word in words]

    if word_vectors:
        text_embedding = np.array(word_vectors)
        vectors = text_embedding.shape[0]
        text_embedding = np.concatenate([text_embedding, np.zeros((text_embeddings-vectors, emb_size))])
        return text_embedding
    else:
        return np.zeros((text_embeddings, emb_size))
    
import torch
import numpy as np

def text_preprocessing(text):
    g_input = []

    for line in text:
        input = embed_sentence(line)
        input = input.astype(float)
        input = input.reshape(-1, emb_size)

        g_input.append(input)

    g_input = np.array(g_input)

    g_input = torch.tensor(g_input)

    return g_input

def getEmbeds(text):
    text = preprocess_text(text)
    embed = embed_sentence(text)
    embed = embed.astype(float)
    return torch.tensor(embed)

import numpy as np
from PIL import Image, ImageFile, ImageOps
import os
import torch
import tensorly as tl
import cv2
from tensorly.decomposition import tucker, parafac
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from torch import nn
import torch.optim as optim
import copy
from sklearn.model_selection import KFold
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(1)

def compute_prnu(image, n=3):
    blur_kernel_size = (n, n)
    blur_sigma = n / 5

    blurred_image = cv2.GaussianBlur(image, blur_kernel_size, blur_sigma)

    noise = (image - blurred_image) ** 2
    noise_sum = np.zeros_like(image, dtype=np.float32)

    for _ in range(n * n):
        noise_sum += noise

    noise_sum = np.sqrt(noise_sum / (n*n))

    # Compute the PRNU by normalizing the noise_sum residual pattern
    prnu = noise_sum / (np.mean(noise_sum) + 1e-10)
    return prnu

    # Compute the PRNU by normalizing the noise residual pattern
    prnu = noise / (np.mean(noise) + 1e-10)
    return prnu

def readImage(imagePath):
  image = Image.open(imagePath)
  image = image.resize((256,256))
  image2 = np.array(image.convert("YCbCr"))[:,:,0]
  
  prnu = compute_prnu(image2)
  prnu = torch.tensor(prnu)
  
  result = torch.stack([prnu], 0)
  return result

class dataset_creation():

  def __init__(self, text, img_path, violent_label, real_label, sentiment_label):
    self.text = text
    self.img_path = img_path
    self.violent_label = violent_label
    self.real_label = real_label
    self.sentiment_label = sentiment_label

  def __len__(self):
    return len(self.img_path)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    sample = {
        "text_embeds" : torch.tensor(getEmbeds(self.text[idx])).float().to(device),
        "images_x" : torch.tensor(readImage(self.img_path[idx])).float().to(device),
        "violent_label" : torch.tensor(self.violent_label[idx]).long().to(device),
        "real_label" : torch.tensor(self.real_label[idx]).long().to(device),
        "sentiment_label" : torch.tensor(self.sentiment_label[idx]).long().to(device)
    }

    return sample
  
train_dataset = dataset_creation(text_train, img_path_train, violent_train, real_train, sentiment_train)
val_dataset = dataset_creation(text_val, img_path_val, violent_val, real_val, sentiment_val)
test_dataset = dataset_creation(text_test, img_path_test, violent_test, real_test, sentiment_test)

from torch.utils.data import DataLoader

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
