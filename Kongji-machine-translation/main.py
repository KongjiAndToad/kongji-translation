import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import time
import os
import easydict
import sys

sys.path.append('/content/drive/MyDrive/kongji-machine-translation')

from data_loader import *
from seq2seq_attn import *
from inference import *
from helper import *
from train import *

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt = easydict.EasyDict({"dialect": "je",
                         "maxlen": 138,
                         "input_level": "syl",
                         "train": True})

# Max Word Length : 30
# Max Sly Length : 138
DIALECT = opt.dialect
INPUT_LEVEL = opt.input_level
train_flag = opt.train

if opt.maxlen == None:
  MAX_LENGTH = 110
else:
    MAX_LENGTH = opt.maxlen


path_je_train = '/content/drive/MyDrive/kongji-machine-translation/ko_je_data.json'
path_je_test = '/content/drive/MyDrive/kongji-machine-translation/ko_je_test.json'

if DIALECT == 'je':
    PATH_TRAIN = path_je_train
    PATH_TEST = path_je_test
else:
    print('Invalid Dialect Error : {DIALECT}')
    exit()

train_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader.readJson(PATH_TRAIN)
test_loader.readJson(PATH_TEST)

SRC = Vocab(train_loader.srcs, INPUT_LEVEL, device)
TRG = Vocab(train_loader.trgs, INPUT_LEVEL, device)
SRC.build_vocab()
TRG.build_vocab()

train_iterator = train_loader.makeIterator(SRC, TRG, sos=True, eos=True)
test_iterator = test_loader.makeIterator(SRC, TRG, sos=True, eos=True)

portion = int(len(test_iterator) * 0.5)
valid_iterator = test_iterator[:portion]
test_iterator = test_iterator[portion:]

INPUT_DIM = SRC.vocab_size
OUTPUT_DIM = TRG.vocab_size
ENC_EMB_DIM = 128 #256
DEC_EMB_DIM = 128 #256
ENC_HID_DIM = 128 #512
DEC_HID_DIM = 128 #512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 24
CLIP = 1

PAD_IDX = TRG.stoi['<pad>']
SOS_IDX = TRG.stoi['<sos>']
EOS_IDX = TRG.stoi['<eos>']

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SOS_IDX, device, MAX_LENGTH).to(device)
## model = nn.DataParallel(model)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

model_name = f's2sAttn_{INPUT_LEVEL}_{DIALECT}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
model_pt_path = f'/content/drive/MyDrive/kongji-machine-translation/models/{model_name}/{model_name}.pt'

print(f'Using cuda : {torch.cuda.get_device_name(0)}')
print(f'Dialect : {DIALECT}')
print(f'Max Length : {MAX_LENGTH}')
print(f'# of train data : {len(train_iterator)}')
print(f'# of test data : {len(test_iterator)}')
print(f'# of valid data : {len(valid_iterator)}')
print(f'SRC Vocab size : {SRC.vocab_size}')
print(f'TRG Vocab size : {TRG.vocab_size}')
print('-' * 20)
print(f'Encoder embedding Dimension : {ENC_EMB_DIM}')
print(f'Decoder embedding Dimension : {DEC_EMB_DIM}')
print(f'Encoder Hidden Dimension : {ENC_HID_DIM}')
print(f'Decoder Hidden Dimension : {DEC_HID_DIM}')
print(f'Encoder dropout rate : {ENC_DROPOUT}')
print(f'Decoder dropout rate : {DEC_DROPOUT}')
print(f'# of epochs : {N_EPOCHS}')
print('-' * 20)
print(f'The model has {count_parameters(model):,} trainable parameters')

try:
    if not os.path.exists(f'/drive/MyDrive/kongji-machine-translation/models/models/{model_name}'):
        os.makedirs(f'/drive/MyDrive/kongji-machine-translation/models/models/{model_name}')
except OSError:
    print(f'Failed to create directory : /drive/MyDrive/kongji-machine-translation/models/models/{model_name}')

if train_flag == True:
    train_model(model = model,
                train_iterator = train_iterator,
                valid_iterator = valid_iterator,
                optimizer = optimizer,
                criterion = criterion,
                CLIP = CLIP,
                N_EPOCHS = N_EPOCHS,
                model_pt_path = model_pt_path)

model.load_state_dict(torch.load(model_pt_path))

##########

src = input()
sentence_list = src.split('.')
result = ""
for sentence in sentence_list:
  result.join(translate_sentence(model, SRC, TRG, sentence, INPUT_LEVEL, device))+".\n"
  print(translate_sentence(model, SRC, TRG, sentence, INPUT_LEVEL, device))
print(result) #문장 입력 받아서 곧장 번역하는 예시
