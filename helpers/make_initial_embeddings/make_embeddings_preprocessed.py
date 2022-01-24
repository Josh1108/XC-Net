import fasttext
import numpy as np
import tables
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import sys
# from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd

p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

from analysis import preprocess, generate_df

modelpath="../../data/COLING/New-FTCondensedData/jobfasttext.bin" # only for fasttext, for bert we load from huggingface
# inputpath='../data/COLING/trn_X.txt'
# testpath='/home/jsk/skill-prediction/data/COLING/tst_X.txt'
testpath = '/home/jsk/skill-prediction/data/COLING/trn_X.txt'
# labelpath='../data/COLING/Y.txt'
# trainembpath="../data/COLING/Fine-FTCondensedData/trn_point_embs.npy"
testembpath="/home/jsk/skill-prediction/data/COLING/New-FT-POSCondensedData/trn_point_embs.npy"
# labelembpath="../data/COLING/Fine-FTCondensedData/label_embs.npy"
# testembpath="/home/jsk/skill-prediction/data/COLING-intermediate/New-FTCondensedData/tst_point_embs.npy"

train = True
model_name='fasttext'
preprocess_type = 'pos-tag'


def make_embs(model,path,vectorfilepath,model_name,dim =300,save_df='/home/jsk/skill-prediction/XC-Net/dumps/df.csv',preprocess_type = preprocess_type,save_file=False):
    file=open(path,"r")
    jobs=file.readlines()
    jobs = [x.split('\n')[0] for x in jobs]
    jobs = preprocess(jobs,preprocess_type)

    if save_file ==True:
        my_file = Path(save_df)
        if not my_file.is_file():
            df = pd.DataFrame(jobs,columns = ['raw_jobs'])
            df.to_csv(save_df,index=None)
        _ = generate_df(save_df,jobs,preprocess_type)
    
    corpus_embeddings = None
    if model_name=='bert' or model_name =='distilbert':
        dim = 768
        corpus_embeddings = np.empty((0, dim))
        for batch in tqdm(np.array_split(jobs, 50)):
            embs = model.encode(batch)
            corpus_embeddings = np.vstack((corpus_embeddings, embs))
    elif model_name=='fasttext':
        dim = 300
        corpus_embeddings = np.empty((0, dim))
        for i in range(len(jobs)):
            words=jobs[i].split()
            if len(words) ==0:
                continue
            vectorlist=[model[word.lower()] for word in words]
            jobvector=np.mean(vectorlist, axis=0)
            corpus_embeddings = np.vstack((corpus_embeddings, jobvector))
    else:
        return NotImplementedError
    print(corpus_embeddings.shape)
    np.save(vectorfilepath,corpus_embeddings)
    return

model = None
if model_name=='bert': 
    h_model_name = "bert-base-nli-mean-tokens"
    model = SentenceTransformer(h_model_name) 
elif model_name == 'distilbert': 
    h_model_name = "distilbert-base-nli-mean-tokens"
    model = SentenceTransformer(h_model_name) 
elif model_name=='fasttext':
    model = fasttext.load_model(modelpath)
if train == True:
    make_embs(model,testpath,testembpath,model_name,save_df = '/home/jsk/skill-prediction/XC-Net/dumps/df_train.csv')
else:
    make_embs(model,testpath,testembpath,model_name,save_file=False)
