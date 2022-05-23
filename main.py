import numpy as np
import json
import argparse
from utils.dataset import KNN_LM, Text2Data
from utils.model import FC
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM
import random
from sklearn.model_selection import KFold
import time
import re
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

# Create the dataset
def post_process(data, train_vec, args, k=3):
    new_data = []
    # Loading the path_list  of paper
    vec_in_train_vec = [train_vec[i][0] for i in range(len(train_vec))]
    vec_in_data = [data[i][0] for i in range(len(data))]
    # print(len(vec_in_train_vec), vec_in_train_vec[0])
    # c = cosine_similarity(vec_in_data,vec_in_train_vec)
    e = euclidean_distances(vec_in_data,vec_in_train_vec)
    # print(e.shape, c.shape) # (1805, 1624)  
    idx_c = np.argsort(e)
    for ct, sample in enumerate(data):
        text = sample[1]
        text_arr = []
        for idx_k in range(k):
            t_idx = idx_c[ct][idx_k+1]
            new_text = train_vec[t_idx][1]
            # t = data[]
            text_arr.append(new_text)
        new_data.append((text,text_arr))
    return new_data

def vec2text(args):
    device_0 = torch.device(args.device)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    raw_data = np.load(args.data_dir + args.platform + '.npy', allow_pickle=True)
    new_raw_data = []
    tau = 0.1
    K = 4
    emb_dim = 768
    emb_dim_2 = 8
    num_heads = 1
    linear1 = nn.Linear(emb_dim, emb_dim_2).to(device_0)
    linear2 = nn.Linear(emb_dim_2,1).to(device_0)
    multihead_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True).to(device_0)

    multihead_attn_2 = nn.MultiheadAttention(emb_dim_2, num_heads, batch_first=True).to(device_0)

    for sample in raw_data:
        new_raw_data.append((sample[0], sample[2]))
    raw_data = np.array(new_raw_data, dtype=object)
    print(len(raw_data))
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Split the dataset first to get the vec space in train_vec
    bleu_1_list = [0] * 10
    bleu_2_list = [0] * 10
    bleu_3_list = [0] * 10
    bleu_4_list = [0] * 10
    rouge_1_list = [0] * 10
    rouge_2_list = [0] * 10
    rouge_L_list = [0] * 10
    meteor_list = [0] * 10
    nist_list = [0] * 10
    for train_index, test_index in kf.split(raw_data):
        print('Fold Starting')
        train_vec = raw_data[train_index]
        # Find the nearest vec to get the source
        data = post_process(raw_data, train_vec, args, k=K) 
        data = np.array(data, dtype=object)
        print('Loading the model')
        model = AutoModelWithLMHead.from_pretrained("t5-base")
        model.to(device_0)
        optimizer = torch.optim.Adam(model.parameters())
        train = data[train_index]
        test = data[test_index]
        train_set = KNN_LM(tokenizer, train)
        test_set = KNN_LM(tokenizer,test)
        train_loader = DataLoader(train_set, batch_size=16, drop_last=True, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=16, drop_last=False, shuffle=True, num_workers=1)
        epoch = 10
        print('Training')
        for i in range(epoch):
            total_loss = 0
            for batch in tqdm(train_loader):
                source = batch['source']
                target = batch['target']
                # input_embs = 0
                input_embs = torch.zeros(len(source[0]), K, args.max_length, emb_dim).to(device_0)
                input_embs_attn = torch.zeros(len(source[0]), K, emb_dim_2).to(device_0)
            
                for k in range(K):
                    source_token = tokenizer.batch_encode_plus(list(source[k]), max_length=args.max_length, add_special_tokens=True, padding='max_length',truncation=True, return_tensors='pt') 
                    input_ids = source_token['input_ids']
                    input_emb = model.get_input_embeddings()(input_ids.to(device_0))
                    # print(input_emb.shape)
                    # sleep
                    attn_output, attn_output_weights = multihead_attn(input_emb, input_emb, input_emb)
                    output = linear1(attn_output[:,0,:])
                    # print(output.shape)
                    input_embs_attn[:, k, :] = output
                    input_embs[:, k, :, :] = input_emb
                    # sleep
                    # input_embs += input_emb
                # input_emb = input_embs / K
                attn_output, attn_output_weights = multihead_attn_2(input_embs_attn, input_embs_attn, input_embs_attn)
                # print(attn_output.shape)
                # attn_output = attn_output.permute(0,2,1)    
                output = linear2(attn_output)
                # output = output.squeeze(2) 
                # print(output, output.shape) # (batch_size, K, 1)
                weight = F.softmax(output*tau, dim=1)
                weight = weight.unsqueeze(-1) 
                # print(weight, weight.shape)
                input_emb = torch.sum(weight*input_embs, dim=1)
                # print(input_emb.shape)
                # sleep
                target_token = tokenizer.batch_encode_plus(target, max_length=args.max_length, add_special_tokens=True, padding='max_length',truncation=True, return_tensors='pt') 
                y_ids = target_token['input_ids']
                y_attention_mask = target_token['attention_mask']
                lm_labels = y_ids.clone()
                lm_labels[y_ids[:,:] == tokenizer.pad_token_id] = -100
                optimizer.zero_grad()
                encoder_outputs = model.get_encoder()(inputs_embeds = input_emb.to(device_0))
                outputs = model(
                    inputs_embeds = input_emb.to(device_0),
                    decoder_attention_mask=y_attention_mask.to(device_0),
                    labels=lm_labels.to(device_0)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss
            print(total_loss/len(train_loader))
            torch.save(model, 'model/t5_{0}'.format(i))
        
        with torch.no_grad():
            print('Short Inferencing')
            max_length = 0
            for i in range(epoch):
                count = 0
                total_bleu_1 = 0
                total_bleu_2 = 0
                total_bleu_3 = 0
                total_bleu_4 = 0
                total_rouge_1 = 0
                total_rouge_2 = 0
                total_rouge_L = 0
                total_meteor = 0
                total_nist = 0
                model = torch.load('model/t5_{0}'.format(i))
                model.to(device_0)
                for batch in tqdm(test_loader):
                    source = batch['source']
                    target = batch['target']
                    max_length = 64
                    input_embs = 0
                    input_embs = torch.zeros(len(source[0]), K, args.max_length, emb_dim).to(device_0)
                    input_embs_attn = torch.zeros(len(source[0]), K, emb_dim_2).to(device_0)
                
                    for k in range(K):
                        source_token = tokenizer.batch_encode_plus(list(source[k]), max_length=args.max_length, add_special_tokens=True, padding='max_length',truncation=True, return_tensors='pt') 
                        input_ids = source_token['input_ids']
                        input_emb = model.get_input_embeddings()(input_ids.to(device_0))
                        attn_output, attn_output_weights = multihead_attn(input_emb, input_emb, input_emb)
                        output = linear1(attn_output[:,0,:])
                        input_embs_attn[:, k, :] = output
                        input_embs[:, k, :, :] = input_emb
                        # input_embs += input_emb
                    # input_emb = input_embs / K
                    attn_output, attn_output_weights = multihead_attn_2(input_embs_attn, input_embs_attn, input_embs_attn)
                    output = linear2(attn_output)
                    weight = F.softmax(output*tau, dim=1)
                    weight = weight.unsqueeze(-1) 
                    input_emb = torch.sum(weight*input_embs, dim=1)

                    target_token = tokenizer.batch_encode_plus(target, max_length=args.max_length, add_special_tokens=True, padding='max_length',truncation=True, return_tensors='pt') 
                    encoder_outputs = model.get_encoder()(inputs_embeds = input_emb.to(device_0))
                    generated_ids = model.generate(
                        encoder_outputs = encoder_outputs,
                        max_length=max_length, 
                        num_beams=2,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
                    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                    for j in range(len(preds)):
                        pred = preds[j].split(' ')
                        gt = target[j].split(' ') 
                        bleu_1 = sentence_bleu([gt], pred, weights=(1,0,0,0))
                        bleu_2 = sentence_bleu([gt], pred, weights=(0,1,0,0))
                        bleu_3 = sentence_bleu([gt], pred, weights=(0,0,1,0))
                        bleu_4 = sentence_bleu([gt], pred, weights=(0,0,0,1))
                        score = scorer.score(target[j], preds[j])
                        rouge_1 = score['rouge1'].recall
                        rouge_2 = score['rouge2'].recall
                        rouge_L = score['rougeL'].recall
                        meteor = meteor_score([target[j]], preds[j])
                        nist = sentence_nist([gt], pred)
                        total_bleu_1 += bleu_1
                        total_bleu_2 += bleu_2
                        total_bleu_3 += bleu_3
                        total_bleu_4 += bleu_4
                        total_rouge_1 += rouge_1
                        total_rouge_2 += rouge_2
                        total_rouge_L += rouge_L
                        total_meteor += meteor
                        total_nist += nist
                        count += 1
                #print(total_bleu_1/count, total_bleu_2/count, total_bleu_3/count, total_bleu_4/count, total_rouge_1/count, total_rouge_2/count, total_rouge_L/count, total_meteor/count, total_nist/count)
                bleu_1_list[i] += total_bleu_1/count
                bleu_2_list[i] += total_bleu_2/count
                bleu_3_list[i] += total_bleu_3/count
                bleu_4_list[i] += total_bleu_4/count
                rouge_1_list[i] += total_rouge_1/count
                rouge_2_list[i] += total_rouge_2/count
                rouge_L_list[i] += total_rouge_L/count
                meteor_list[i] += total_meteor/count
                nist_list[i] += total_nist/count
                torch.cuda.empty_cache()
    bleu_1_list = np.array(bleu_1_list)
    bleu_2_list = np.array(bleu_2_list)
    bleu_3_list = np.array(bleu_3_list)
    bleu_4_list = np.array(bleu_4_list)
    rouge_1_list = np.array(rouge_1_list)
    rouge_2_list = np.array(rouge_2_list)
    rouge_L_list = np.array(rouge_L_list)
    meteor_list = np.array(meteor_list)
    nist_list = np.array(nist_list)
    print('bleu_1: ', bleu_1_list/5)
    print('bleu_2: ', bleu_2_list/5)
    print('bleu_3: ', bleu_3_list/5)
    print('bleu_4: ', bleu_4_list/5)
    print('rouge_1: ', rouge_1_list/5)
    print('rouge_2: ', rouge_2_list/5)
    print('rouge_L: ', rouge_L_list/5)
    print('meteor: ', meteor_list/5)
    print('nist: ', nist_list/5)

def text2vec(args):
    device_0 = torch.device(args.device)
    raw_data = np.load(args.data_dir + args.platform + '.npy', allow_pickle=True)  
    new_raw_data = []
    for sample in raw_data:
        vec = sample[0]
        embedding = sample[3]
        new_raw_data.append((vec,embedding))
    raw_data = np.array(new_raw_data, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    corr_list = [0] * 20
    for train_index, test_index in kf.split(raw_data):
        model = FC(768,100000)
        model.to(device_0)
        criteria = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        train = raw_data[train_index]
        test = raw_data[test_index]
        train_set = Text2Data(train)
        test_set = Text2Data(test)
        train_loader = DataLoader(train_set, batch_size=16, drop_last=True, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=16, drop_last=False, shuffle=True, num_workers=1)
        epoch = 20
        for i in range(epoch):
            total_loss = 0
            for batch in tqdm(train_loader):
                embed = batch['emb'].to(device_0)
                vec = batch['vec'].float()
                label = vec.to(device_0)
                output = model(embed)
                loss = torch.sqrt(criteria(output, label))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            avg_loss = total_loss / len(train_loader)
            print('Epoch: ' + str(i) + ' with avg_loss ' + str(avg_loss))
            torch.save(model, 'model/text2data_{0}'.format(i))
        with torch.no_grad():
            for i in range(epoch):
                total_corr = 0
                count = 0
                model = torch.load('model/text2data_{0}'.format(i))
                model.to(device_0)
                for batch in tqdm(test_loader):
                    embed = batch['emb'].to(device_0)
                    vec = batch['vec'].float()
                    label = vec.to(device_0)
                    output = model(embed)
                    loss = torch.sqrt(criteria(output, label))
                    total_corr += loss.item()
                    count += 1
                print(total_corr/count)
                corr_list[i] += total_corr/count
    corr_list = np.array(corr_list)
    print(corr_list/5)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default= 'data/', type=str)
    parser.add_argument("--log_dir", default='result/board', type=str)    
    parser.add_argument("--device", type=str, default='cuda:0', help='device') 
    parser.add_argument("--emb_dim", type=int, default=768, help='specter dim') 
    parser.add_argument("--max_length", type=int, default=64, help='max_length')  
    parser.add_argument("--top-k", type=int, default=5, help='top_k')
    parser.add_argument("--platform", type=str, default='GPL570', help='platform')
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    vec2text(args)
    text2vec(args)