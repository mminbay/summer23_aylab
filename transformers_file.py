# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv('/home/aghasemi/ukbb_v2-master/Transformers/Data/filtered_final_dataset.csv', index_col=0)

# Randomly shuffle the train and test dataframes
train_df = shuffle(df, random_state=1)

# Create lists for train set
train_strings = []
train_phq9_values = []

for index, row in train_df.iterrows():
    # Concatenate all column values except PHQ9_binary
    string = ' '.join([str(row[col]) for col in train_df.columns if col != 'PHQ9_binary'])
    train_strings.append(string)
    
    # Append PHQ9 value to train_phq9_values
    train_phq9_values.append(row['PHQ9_binary'])
 
# Create train dataframe
train_df = pd.DataFrame({'sentence': train_strings, 'label': train_phq9_values})

# Save the DataFrames to a CSV file
train_df.to_csv('/home/aghasemi/ukbb_v2-master/Transformers/train_dataset_real2.csv', index=False)

print("Train DataFrame:")
print(len(train_df))

# train_df['label'].hist()

from datasets import load_dataset
raw_dataset = load_dataset('csv', data_files = '/home/aghasemi/ukbb_v2-master/Transformers/train_dataset_real2.csv')
raw_dataset
split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)

split

# Import AutoTokenizer and create tokenizer object
from transformers import AutoTokenizer
checkpoint = 'bert-base-cased'
tokernizer = AutoTokenizer.from_pretrained(checkpoint, do_basic_tokenize=True)

def tokenize_fn(batch):
  return tokernizer(batch['sentence'], truncation=True)
tokenized_dataset = split.map(tokenize_fn, batched=True)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

from torchinfo import summary
summary(model)

training_args = TrainingArguments(
                                  
    output_dir='training_dir',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_steps=1,  # Adjust the logging frequency if desired
    logging_dir = './logs'
)

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'micro')
    return {'accuracy': acc, 'f1_score': f1}

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["test"],
    tokenizer=tokernizer,
    compute_metrics=compute_metrics
)

trainer.train()

! ls training_dir

from transformers import pipeline

saved_model = pipeline('text-classification', model='training_dir/checkpoint-132')
split['test']
predictions = saved_model(split['test']['sentence'])

predictions[:10]

def get_label(d):
  return int(d['label'].split('_')[1])
predictions = [get_label(d) for d in predictions]

print("acc:",accuracy_score(split['test']['label'], predictions))
print("f1:",f1_score(split['test']['label'], predictions, average = 'macro'))

# create function for plotting confusion matrix
def plot_cm(cm):
  classes = ['depressed','not depressed']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sns.heatmap(df_cm, annot = True, fmt='g')
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')

cm = confusion_matrix(split['test']['label'],predictions, normalize = 'true')
plot_cm(cm)

### Attention Weights Visualization

### Error Analysis: Analyzing BERT's attention

import torch
from transformers import *
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer

plt.figure(figsize = (20, 20))

BERT_PATH = "training_dir/checkpoint-132"
model = BertModel.from_pretrained(BERT_PATH, output_attentions = True)
print('Model loaded.')

#### Code #1

!pip install ansi
from typing import Union

import numpy as np
import seaborn as sns
from ansi.colour import rgb

def color_text(text, rgb_code):
    reset =  '\x1b[0m'
    return rgb.rgb256(*rgb_code) + text + reset

def value2rgb(value):
#     if value < 0:
#         rgb_code = (255/2 + abs(value)/2, abs(value), 255/2 + abs(value)/2)
#     else:
#         rgb_code = (125+value/2, 0, 255/2-value/2)
    if value < 0:
        rgb_code = (255, 255, abs(value))
    else:
        rgb_code = (255, 255-value, 0)
    return rgb_code

def scale(values, input_range, output_range):
    return np.interp(values, input_range, output_range)

def get_legends(value_range, scale_to, step=5):
    min_value, max_value = value_range
    leg_values = np.linspace(min_value, max_value, step)
    scaled_values = scale(leg_values, (min_value, max_value), scale_to)
    
    legends = []
    for leg_value, scaled_value in zip(leg_values, scaled_values):
         legends.append(color_text('{:.2f}'.format(leg_value), value2rgb(scaled_value)))
    return legends

def color_texts(texts, values, use_absolute):
    if use_absolute:
        value_range = (0, 1)
    else:
        value_range = (min(values), max(values))
    scale_to = (-255, 255)
    scaled_values = scale(values, value_range, scale_to)
    result = []
    for text, value in zip(texts, scaled_values):
        rgb = value2rgb(value)
        result.append(color_text(text, rgb))
    colored = ' '.join(result)
    legends = get_legends(value_range, scale_to)

    colored += ' ({})'.format(' '.join(legends))
        
    if use_absolute:
        colored += ' (min: {:.10f} max: {:.10f})'.format(min(values), max(values))
    return colored

def visual_matrix(matrix, labels=None, title=None, **kwargs):
    sns.set()
    ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, **kwargs)
    if title:
        ax.set(title = title)
#     ax.xaxis.tick_top()
    return ax


def get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns):
    if layer_num is None:
        layer_num = -1  # last layer
    
    batch_size = len(atns[0])
    if batch_size == 1:
        batch_num = 0
    else:
        if batch_num is None:
            raise ValueError('You input an attention with batch size != 1. Please input attentions with batch size 1 or specify the batch_num you want to visualize.')
            
    if head_num is None:
        head_num = 'average'

    if token_num is None:
        token_num = 'average'

    if atn_axis is None:
        atn_axis = 0
        
    return layer_num, batch_num, head_num, token_num, atn_axis


def get_multihead_atn_matrix(atns, layer_num=None, batch_num=None):
    
    
#     layer_num, batch_num = get_or_default_layer_and_batch_num(layer_num, batch_num, atns)
    layer = atns[layer_num]
    try:
        multihead_atn_matrix = layer[batch_num].detach().numpy()  # pytorch
    except TypeError:
        multihead_atn_matrix = layer[batch_num].cpu().numpy()  # pytorch
    except AttributeError:
        multihead_atn_matrix = layer[batch_num]  # tensorflow
    return multihead_atn_matrix

def get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num):
    # atn_matrix: (sequence_length, sequence_length)       
    try:
        atn_matrix = multihead_atn_matrix[head_num]
    except (IndexError, TypeError):
        # average over heads        
        atn_matrix = np.mean(multihead_atn_matrix, axis=0)
    return atn_matrix

def merge_atn_matrix(atn_matrix, mean_over_mat_axis):
    atn_matrix_over_axis: list = np.mean(atn_matrix, axis=mean_over_mat_axis)
    return atn_matrix_over_axis

def matrix2values(matrix, index='average', axis=0):    
    if index == 'average':
        result_mat = np.mean(matrix, axis=axis)
    elif isinstance(index, int):
        if axis == 0:
            result_mat = matrix[index]
        elif axis == 1:
            result_mat = matrix.T[index]
        else:
            raise ValueError('matrix to values have a wrong axis (0 or 1): ' + str(axis))
    else:
        raise ValueError('matrix to values have a wrong index ("average" or integers): ' + str(index))
    return result_mat
        
def get_atn_values(layer_num, batch_num, head_num, token_num, atn_axis, atns):
    layer_num, batch_num, head_num, token_num, atn_axis = get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns)
    multihead_atn_matrix = get_multihead_atn_matrix(atns, layer_num=layer_num, batch_num=batch_num)
    atn_matrix = get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num=head_num)
    atn_values = matrix2values(atn_matrix, index=token_num, axis=atn_axis)
    
    return atn_values

def get_atn_matrix(layer_num, batch_num, head_num, atns):
    layer_num, batch_num, head_num, *_ = get_or_default_config(layer_num, batch_num, head_num, None, None, atns)

    multihead_atn_matrix = get_multihead_atn_matrix(atns, layer_num=layer_num, batch_num=batch_num)
    atn_matrix = get_atn_matrix_from_mh_matrix(multihead_atn_matrix, head_num=head_num)
    return atn_matrix

def visual_atn(labels, atns, layer_num=None, batch_num=None, head_num=None, token_num=None, atn_axis=None,
               use_absolute=False, output=False, **kwargs):
    atn_values = get_atn_values(layer_num, batch_num, head_num, token_num, atn_axis, atns)
    layer_num, batch_num, head_num, token_num, atn_axis = get_or_default_config(layer_num, batch_num, head_num, token_num, atn_axis, atns)

    assert len(labels) == len(atn_values), 'len(labels): {}, len(merged_atn_values): {}'.format(len(labels), len(atn_values))

    colored = color_texts(labels, atn_values, use_absolute)

    try:
        label = labels[token_num]
    except TypeError:
        label = 'ALL_TOKENS'

    print('(layer) {} (batch) {} (head) {} (token_num) {} (token) {} (axis) {}'.format(layer_num, batch_num, head_num, token_num, label, atn_axis))

    if output:
        return colored, atn_values
    else:
        return colored

def visual_atn_matrix(labels, atns, layer_num=None, batch_num=None, head_num=None, token_num=None, output=False) -> 'Axes':    
    atn_matrix = get_atn_matrix(layer_num, batch_num, head_num, atns)   
    layer_num, batch_num, head_num, token_num, _ = get_or_default_config(layer_num, batch_num, head_num, token_num, None, atns)
    title = '(layer) {} (batch) {} (head) {}'.format(layer_num, batch_num, head_num)
    
    if output:
        return visual_matrix(atn_matrix, labels, title=title), atn_matrix
    else:
        return visual_matrix(atn_matrix, labels, title=title)

### Token to head matrices

from transformers import BertTokenizer

source_code = train_df.iloc[0]['sentence']

tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_basic_tokenize=False)
input_ids = [tokenizer.encode(source_code)]
tokens = [tokenizer.decode(id_) for id_ in input_ids[0]]

print(tokens)

outputs = model(torch.tensor(input_ids))
loss, logits, attentions = outputs
attentions = outputs['attentions']
print('Attention extracted.')

print(visual_atn(tokens, attentions))
visual_atn_matrix(tokens, attentions)
plt.show()

### Attribution Scores of all layers

visual_atn_matrix(tokens, attentions, layer_num=-1, head_num='average')  # last layer, average over multi-head attention matrices
plt.show()

### Attention Heatmaps

import pandas as pd
from underthesea import word_tokenize, sent_tokenize, text_normalize
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import  LayerIntegratedGradients, visualization as viz, LayerConductance
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


device = 'cpu'
print(device)

df = pd.read_csv('/home/aghasemi/ukbb_v2-master/Transformers/train_dataset_real.csv')

# labels
labels = df['label'].unique().tolist()
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}
print(f'label: {id2label}')

df['labels'] = df['label'].map(label2id)
df.drop(columns=['label'], inplace=True)

def apply_word_tokenize(sen):
    sen = " ".join(sen.split())
    sens = sent_tokenize(sen)
    tokenized_sen = []
    for sen in sens:
        tokenized_sen += word_tokenize(text_normalize(sen))
    return ' '.join([' '.join(words.split(' ')) for words in tokenized_sen])


df['token'] = df['sentence'].map(lambda x: apply_word_tokenize(x.upper()))
df.drop(columns=['sentence'], inplace=True)
df.head()

pretrain_name = "training_dir/checkpoint-132"
tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
model = AutoModelForSequenceClassification.from_pretrained(pretrain_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
model.to(device)

model.bert.embeddings.word_embeddings.weight.shape

model.bert.embeddings.position_embeddings.weight.shape

model.bert.embeddings.token_type_embeddings

class XAI:
    def __init__(self, text_, label_, tokenizer_, model_, device_):
        self.text = text_
        self.label = label_
        self.tokenizer = tokenizer_
        self.model = model_
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.device = device_
        self.input_ids = None
        self.ref_input_ids = None

    def construct_input_ref(self):
        text_ids = self.tokenizer.encode(self.text, add_special_tokens=False)
        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(text_ids) + [self.sep_token_id]

        self.input_ids = torch.tensor([input_ids], device=device)
        self.ref_input_ids = torch.tensor([ref_input_ids], device=device)

        return self.input_ids, self.ref_input_ids

    def custom_forward(self, inputs):
        # return torch.softmax(self.model(inputs)[0], dim=1)[0]  # multi-class
        return torch.sigmoid(self.model(inputs)[0])[0]  # binary

    def visualize(self):
        self.input_ids, self.ref_input_ids = self.construct_input_ref()
        self.all_tokens = tokenizer.convert_ids_to_tokens(self.input_ids[0])

        lig = LayerIntegratedGradients(self.custom_forward, self.model.bert.embeddings)
        attributions, delta = lig.attribute(inputs=self.input_ids,
                                            baselines=self.ref_input_ids,
                                            n_steps=500,
                                            internal_batch_size=3,
                                            return_convergence_delta=True)

        attributions = attributions.sum(dim=-1).squeeze()
        attributions_sum = attributions / torch.norm(attributions)

        score_bert = self.custom_forward(self.input_ids)
        prod_pred = score_bert.max()
        class_pred = score_bert.argmax()

        print(f'{Colors.OKCYAN}Text:{Colors.ENDC} {text} \n'
              f'{Colors.OKCYAN}Predicted Probability:{Colors.ENDC} {prod_pred:,.2f}\n'
              f'{Colors.OKCYAN}Predicted Class:{Colors.ENDC} {class_pred} '
              f'({id2label[class_pred.item()]}) vs. True Class: {self.label} ({id2label[self.label]})')

        score_vis = viz.VisualizationDataRecord(attributions_sum,
                                                pred_prob=prod_pred,
                                                pred_class=class_pred,
                                                true_class=self.label,
                                                attr_class=class_pred,
                                                attr_score=attributions_sum.sum(),
                                                raw_input_ids=self.all_tokens,
                                                convergence_score=delta)

        viz.visualize_text([score_vis])
        return attributions_sum

        
    def get_topk_attributed_tokens(self, attrs, k=5):
        values, indices = torch.topk(attrs, k)
        top_tokens = [self.all_tokens[idx] for idx in indices]
        return pd.DataFrame({'Word': top_tokens, 'Index': indices, 'Attribution': values})

report = pd.DataFrame()
for i in [1, 3, 4]:
    text = df['token'].values[i]
    label = df['labels'].values[i]
    explain = XAI(text, label, tokenizer, model, device)
    attributions_sum = explain.visualize()
    
    df_topk = explain.get_topk_attributed_tokens(attributions_sum)
    df_topk['Text'] = text
    report = pd.concat([report, df_topk])
    
    print(10*'=')

### 2.2 LayerConductance - Hidden Layers

def forward_func2(inputs):
    return model(inputs_embeds=inputs)[0].max(1).values

sep_token_id = tokenizer.sep_token_id
cls_token_id = tokenizer.cls_token_id

text_ids = tokenizer.encode(text, add_special_tokens=False)
input_ids = [cls_token_id] + text_ids + [sep_token_id]
input_ids = torch.tensor([input_ids], device=device)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

layer_attr = []
layer_attr_dist = defaultdict(list)

interpretable_embedding = model.bert.embeddings.word_embeddings
input_embeddings = interpretable_embedding(input_ids)

num_hidden_layers = model.bert.config.num_hidden_layers
for i in tqdm(range(num_hidden_layers)):
    lc = LayerConductance(forward_func2, model.bert.encoder.layer[i])
    lc_vals = lc.attribute(input_embeddings)
    
    lc_norm = lc_vals.sum(dim=-1).squeeze(0)
    lc_norm = lc_norm / torch.norm(lc_norm)
    lc_norm = lc_norm.cpu().tolist()
    
    layer_attr.append(lc_norm)
    for idx, select_token in enumerate(all_tokens):
        if len(layer_attr_dist[select_token]) == num_hidden_layers:
            pass
        else:
            layer_attr_dist[select_token].append(lc_vals[0, idx, :].cpu().detach().tolist())

indices = input_ids[0].tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)
data = pd.DataFrame(layer_attr, columns=all_tokens)

plt.figure(figsize=(25, 6))
sns.heatmap(data[data > 0], linewidth=0.2, annot=True, cmap = sns.diverging_palette(230, 20, as_cmap=True), fmt=',.2f')
plt.ylabel('Hidden Layers')
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
xticklabels=all_tokens
yticklabels=list(range(1,13))
ax = sns.heatmap(torch.tensor(layer_attr).detach().cpu().numpy(), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
plt.xlabel('Tokens')
plt.ylabel('Layers')