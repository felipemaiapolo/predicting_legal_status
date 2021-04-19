import copy
import ftfy
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import random
from random import choice
import re
import seaborn as sns
import tensorflow as tf
import time
from tqdm import tqdm
import unidecode

import xgboost as xgb

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim.similarities.index import AnnoyIndexer
from gensim.test.utils import common_texts

from IPython.display import SVG
from itertools import product

import tensorflow
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Input, Flatten, LSTM, Conv1D, Dense, TimeDistributed, GlobalMaxPooling1D, Lambda
from tensorflow.keras.models import Model, load_model, Sequential

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from transformers import Trainer, TrainingArguments