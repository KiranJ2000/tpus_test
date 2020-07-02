
import os
assert os.environ['COLAB_TPU_ADDR']

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

!export XLA_USE_BF16=1

!pip install transformers

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import transformers
import torch
import torch.nn as nn
import config_bert
import tqdm

from sklearn.model_selection import train_test_split
from transformers import AdamW,get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
from sklearn import metrics
# %matplotlib inline

df = pd.read_csv('sentiment.csv')
df.head()

df.drop('Unnamed: 0',axis=1,inplace=True)
df.info()

df = df[df['review'].str.split().map(lambda x: len(x))<200]
df.info()

class BertDataset:
  def __init__(self,text,target):
    self.text = text
    self.target = target
    self.tokenizer = config_bert.TOKENIZER
    self.max_len = config_bert.MAX_LEN

  def __len__(self):
    return len(self.text)

  def __getitem__(self,item):
    text = str(self.text[item])
    targets = self.target[item]

    inputs = self.tokenizer.encode_plus(
                 text,
                 None,
                 return_attention_mask = True,
                 return_token_type_ids = True,
                 pad_to_max_length = True,
                 max_length = self.max_len,
                 truncation = True
                              )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return {
        'ids': torch.tensor(ids,dtype=torch.long),
        'mask': torch.tensor(mask,dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids,dtype=torch.long),
        'targets': torch.tensor(targets,dtype=torch.float)
    }

class BertPTModel(nn.Module):
  def __init__(self):
    super(BertPTModel,self).__init__()
    self.bert = transformers.BertModel.from_pretrained(config_bert.BERT_PATH)
    self.dropout = nn.Dropout(0.3)
    self.linear = nn.Linear(768,1)

  def forward(self,ids,mask,token_type_ids):
    _,o2 = self.bert(ids,
                     attention_mask = mask,
                     token_type_ids = token_type_ids
                    )
    return self.linear(self.dropout(o2))

def loss_fn(outputs,targets):
  return nn.BCEWithLogitsLoss()(outputs,targets.view(-1,1))

def train_fn(data_loader,optimizer,model,device,shedular):
  model.train()
  losses = []

  for bi,d in tqdm.notebook.tqdm(enumerate(data_loader),total=len(data_loader)):
    ids = d["ids"]
    mask = d["mask"]
    token_type_ids = d["token_type_ids"]
    targets = d["targets"]

    ids = ids.to(device,dtype=torch.long)
    mask = mask.to(device,dtype=torch.long)
    token_type_ids = token_type_ids.to(device,dtype=torch.long)
    targets = targets.to(device,dtype=torch.float)

    optimizer.zero_grad()
    outputs = model(ids = ids,
                    mask = mask,
                    token_type_ids=token_type_ids)
    loss = loss_fn(outputs,targets)
    losses.append(loss.item())
    loss.backward()
    xm.optimizer_step(optimizer,barrier=True)
    shedular.step()
  return np.mean(losses)


def eval_fn(data_loader,model,device):
  model.eval()
  loss1 = []
  fin_targets = []
  fin_outputs = []
  for bi,d in tqdm.notebook.tqdm(enumerate(data_loader),total=len(data_loader)):
    ids = d["ids"]
    mask = d["mask"]
    token_type_ids = d["token_type_ids"]
    targets = d["targets"]

    ids = ids.to(device,dtype=torch.long)
    mask = mask.to(device,dtype=torch.long)
    token_type_ids = token_type_ids.to(device,dtype=torch.long)
    targets = targets.to(device,dtype=torch.float)

    outputs = model(ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)
    loss = loss_fn(outputs, targets)
    loss1.append(loss.item())

    fin_targets.extend(targets.cpu().detach().numpy().tolist())
    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

  return fin_outputs,fin_targets,np.mean(loss1)

def run():
  df["sentiment"] = df["sentiment"].apply(lambda x: 0 if x == "negative" else 1)
  df_train,df_valid = train_test_split(df,test_size=0.1)

  df_train = df_train.reset_index(drop=True)
  df_valid = df_valid.reset_index(drop=True)

  train_dataset = BertDataset(df_train['review'].values,
                              df_train['sentiment'].values)
  train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=config_bert.BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=4)
  
  valid_dataset = BertDataset(df_valid['review'].values,
                              df_valid['sentiment'].values)
  valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                  batch_size=config_bert.VALID_BATCH_SIZE,
                                                  num_workers=1)
  
  model = BertPTModel()
  device = xm.xla_device()
  model = model.to(device)

  param_optimizer = list(model.named_parameters())
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
  
  number_of_training_steps = int(len(df_train)/config_bert.BATCH_SIZE * config_bert.EPOCHS)
  optimizer = AdamW(optimizer_parameters,
                    lr=3e-5)
  shedular = get_linear_schedule_with_warmup(optimizer,
                                             num_warmup_steps=0,
                                             num_training_steps=number_of_training_steps)
  
  best_accuracy = 0
  for epoch in range(config_bert.EPOCHS):
    xm.master_print(f'Epoch { epoch + 1}/{config_bert.EPOCHS}')
    xm.master_print('-'*10)
    train_loss = train_fn(train_data_loader,optimizer,model,device,shedular)
    print('\n')
    print(f'Train loss: {train_loss}')
    outputs,targets,loss_eval = eval_fn(valid_data_loader,model,device)
    print(f'Eval loss {loss_eval}')
    outputs = np.array(outputs) >=0.5
    accuracy = metrics.accuracy_score(targets,outputs)
    print('Accuracy = ',accuracy)
    if accuracy > best_accuracy:
      xm.save(model.state_dict(),'checkpoint'+str(epoch)+'.pth')
      best_accuracy = accuracy

run()

def predict(sentence,model,device):
  senti = ['negative','positive']
  model.eval()
  inputs = config_bert.TOKENIZER.encode_plus(sentence,
                                             None,
                                             return_attention_mask = True,
                                             add_special_tokens=True,
                                             return_token_type_ids=True,
                                             pad_to_max_length=True,
                                             max_length = config_bert.MAX_LEN,
                                             truncation=True,
                                             return_tensors='pt')
  
  #ids = inputs["input_ids"]
  #mask = inputs["attention_mask"]
  #token_type_ids = inputs["token_type_ids"]

  ids = inputs['input_ids'].to(device)
  mask = inputs['attention_mask'].to(device)
  token_type_ids = inputs['token_type_ids'].to(device)
    
  output = model(ids,mask,token_type_ids)
  #prediction = torch.sigmoid(output).cpu().detach().numpy().tolist()
  return senti[int(np.round(nn.Sigmoid()(output.detach().cpu()).item()))]

model = BertPTModel()
device = xm.xla_device()
model = model.to(device)
model.load_state_dict(torch.load('checkpoint2.pth'))
predict('Good job',model,device)