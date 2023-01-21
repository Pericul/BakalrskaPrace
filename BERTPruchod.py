# Import knihoven
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import trange
import random

# Nahrani excelu
df = pd.read_excel(r"C:\Users\jirih\Desktop\Dulezite\3.BERT\AnotovanaData.xlsx")
text = df.text.values
labels = df.label.values

# Tokenizace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
token_id = []
attention_masks = []
def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )


for sample in text:
  encoding_dict = preprocessing(sample, tokenizer)
  token_id.append(encoding_dict['input_ids']) 
  attention_masks.append(encoding_dict['attention_mask'])

token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)

# Nastavení proměných modelu
val_ratio = 0.2
batch_size = 16

# Rozdělení dat na trénovací a ověřovací skupiny
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

train_set = TensorDataset(token_id[train_idx], 
                          attention_masks[train_idx], 
                          labels[train_idx])

val_set = TensorDataset(token_id[val_idx], 
                        attention_masks[val_idx], 
                        labels[val_idx])

train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )

# Definice parametrů měřících výkonost modelu
# True Positives (TP): Správné předpovědi labelu 1
def b_tp(preds, labels):
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

# False Positives (FP): Špatné předpovědi labelu 1
def b_fp(preds, labels):
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

# True Negatives (TN): Správné předpovědi labelu 0
def b_tn(preds, labels):
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

# False Negatives (FN): Špatné předpovědi labelu 0
def b_fn(preds, labels):
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


# Jednotlivá měření:
#   - Přesnost        = (TP + TN) / N
#   - Preciznost      = TP / (TP + FP)
#   - Rozpoznání      = TP / (TP + FN)
#   - Specifičnost    = TN / (TN + FP)
def b_metrics(preds, labels):
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

  
# Načtení modelu BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

# RNastavení jednotlivých parametrů pro učení modelu
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )
epochs = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for _ in trange(epochs, desc = 'Epoch'):   
    # Nastavení modelu na TRÉNOVÁNÍ
    model.train()
    
    # Trening
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        train_output = model(b_input_ids, 
                             token_type_ids = None, 
                             attention_mask = b_input_mask, 
                             labels = b_labels)
        train_output.loss.backward()
        optimizer.step()
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    # Nastavení modelu na VALIDOVÁNÍ
    model.eval()
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Výpočet výkonostních parametrů
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        if b_precision != 'nan': val_precision.append(b_precision)
        if b_recall != 'nan': val_recall.append(b_recall)
        if b_specificity != 'nan': val_specificity.append(b_specificity)


################################
#####  Predikce- Ověření   #####
################################

# Načtení označeného datového souboru, pro získání informace o spolehlivosti natrénovaného modelu 
df2 = pd.read_excel(r"C:\Users\jirih\Desktop\Dulezite\3.BERT\AnotovanaData.xlsx")
df2['Propaganda'] = 'Placeholder'
df2['Neni'] = 'Placeholder'
df2['Je'] = 'Placeholder'

for ind in df2.index:
    test_ids = []
    test_attention_mask = []
    sentenc = df2['text'][ind]

    encoding = preprocessing(sentenc, tokenizer)

    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    with torch.no_grad():
      output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 1 if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 0

    # Uložení výsledků do tabulky 
    df2['Propaganda'][ind] = prediction
    df2['Je'][ind] = output.logits.cpu().numpy()[0][0]
    df2['Neni'][ind] = output.logits.cpu().numpy()[0][1]

df2.to_excel("C:\\Users\\jirih\\Desktop\\Dulezite\\3.BERT\\AnotovanaDataKontrola.xlsx")


###############################
#####  Predikce - Twitter #####
###############################

# Načtení datového souboru získaného z TWitteru. 
df2 = pd.read_excel(r"C:\Users\jirih\Desktop\Dulezite\3.BERT\Twitter.xlsx")
df2['Propaganda'] = 'Placeholder'
df2['Neni'] = 'Placeholder'
df2['Je'] = 'Placeholder'

for ind in df2.index:
    test_ids = []
    test_attention_mask = []
    sentenc = df2['text'][ind]

    encoding = preprocessing(sentenc, tokenizer)

    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    with torch.no_grad():
      output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 1 if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 0

    # Uložení výsledků do tabulky 
    df2['Propaganda'][ind] = prediction
    df2['Je'][ind] = output.logits.cpu().numpy()[0][0]
    df2['Neni'][ind] = output.logits.cpu().numpy()[0][1]

df2.to_excel("C:\\Users\\jirih\\Desktop\\Dulezite\\3.BERT\\TwitterSPredikci.xlsx")

###############################
##### Predikce - Telegram #####
###############################

# Načtení datového souboru získaného z TWitteru. 
df2 = pd.read_excel(r"C:\Users\jirih\Desktop\Dulezite\3.BERT\Telegram.xlsx")
df2['Propaganda'] = 'Placeholder'
df2['Neni'] = 'Placeholder'
df2['Je'] = 'Placeholder'

for ind in df2.index:
    test_ids = []
    test_attention_mask = []
    sentenc = df2['text'][ind]

    encoding = preprocessing(sentenc, tokenizer)

    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    with torch.no_grad():
      output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 1 if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 0

    # Uložení výsledků do tabulky 
    df2['Propaganda'][ind] = prediction
    df2['Je'][ind] = output.logits.cpu().numpy()[0][0]
    df2['Neni'][ind] = output.logits.cpu().numpy()[0][1]

df2.to_excel("C:\\Users\\jirih\\Desktop\\Dulezite\\3.BERT\\TelegramSPredikci.xlsx")
