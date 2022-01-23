from os import XATTR_CREATE
import scipy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
import ast
import argparse
# import tsne
from xclib.data import data_utils
from scipy import sparse
import plotly
import plotly.express as px
import plotly.io as pio
from numpy import dot
from numpy.linalg import norm
pio.renderers.default = "vscode"


def load_csr(corpus_name: str,emb2):
    print("test",corpus_name.rsplit('.',1)[1])
    if corpus_name.rsplit('.',1)[1] =='npz': # Not being used at the moment
        corpus = sparse.load_npz(corpus_name).toarray()
        index=[np.where(corpus[i]==1)[0] for i in range(corpus.shape[0])]
        emb2_all=np.array([[np.array(emb2[k]) for k in j] for j in enumerate(index)])

    elif corpus_name.rsplit('.',1)[1] =='predictions':
        f = open(corpus_name,'r')
        index =[eval(x.split('\n')[0]) for x in f.readlines()]
        emb2_all =np.array([[np.array(emb2[k]) for k in j] for j in index])

    else:
        corpus = data_utils.read_sparse_file(corpus_name,force_header =True).toarray()
        index=[np.where(corpus[i]==1)[0] for i in range(corpus.shape[0])]
        emb2_all=np.array([[np.array(emb2[k]) for k in j] for j in index])

    return emb2_all

def calculate_cosine(path1,path2,inputfile):
    """
    top k labels. comparison. Heatmap b/w job and k labels. store matrix
    """
    emb1=np.load(path1,allow_pickle=True)
    emb2=np.load(path2,allow_pickle=True)
    for i in range(emb2.shape[0]):
        if norm (emb2[i,:])==0:
            print("index",i)
    print("embedding labels",emb2.shape,emb1.shape)
    emb2_all = load_csr(inputfile,emb2)
    print("label embedding array shape",emb2.shape,emb2[1].shape)
    print("label embedding all array shape",emb2_all.shape,emb2_all[1].shape)
    cosine_sim =[]
    a =1
    for j in range(emb1.shape[0]):
        lbl_cs=[]
        for i in range(20):
            emb_lbl = emb2_all[j,i,:]
            emb_jd = emb1[j]
            # print(emb_lbl,emb_jd)
            
            if norm(emb_lbl)==0:
                a+=1
                print("label embedding",emb_lbl)
                print("counter",a)
            cos_sim = dot(emb_lbl,emb_jd)/(norm(emb_lbl)*norm(emb_jd))
            cosine_sim.append(cos_sim)
    print("cosine_sim shape",len(cosine_sim),cosine_sim[0])
    quit()
    return cosine_sim

def avg_calculate_cosine(path1,path2,inputfile):
    """
    2k*2k
    """
    emb1=np.load(path1,allow_pickle=True) # (no_jobs,hidden_size)
    emb2=np.load(path2,allow_pickle=True) # labels

    emb2_all = load_csr(inputfile,emb2) 
   
    print("label embedding array shape",emb2.shape,emb2[1].shape)
    print("label embedding array shape",emb2_all.shape,emb2_all[1].shape)
    #(no.of jobs,embedding_hidden)
    average_emb2=np.mean(emb2_all,axis=1) # avg labels
    print("average embedding array shape",average_emb2.shape)
    print("average embedding array shape",emb1.shape[0],emb1.shape,emb1[0].shape)
    for m in range(emb1.shape[0]): #ASK NG?
         cosine_sim=cosine(emb1,average_emb2)
    print(cosine_sim.shape)
    return cosine_sim

def euclidiean():

    return NotImplementedError

def topkvalues(cosine_sim_matrix,k=10):
    cosine_sim_ind = []
    cosine_sim_values=[]
    for i in [np.argsort(row) for row in cosine_sim_matrix]:
        cosine_sim_ind.append(i[-k:][::-1])
    for i in [np.sort(row) for row in cosine_sim_matrix]:
        cosine_sim_values.append(i[-k:][::-1])
    print("shape of cosine similarity matrix",len(cosine_sim_ind),len(cosine_sim_values),cosine_sim_ind[0],cosine_sim_values[0])
    return cosine_sim_ind,cosine_sim_values

def visualize(cosine_sim_values):

    fig = px.imshow(np.asarray(cosine_sim_values[:20]),labels=dict(x="Average label embedding", y="avg. document embedding", color="cosine similarity"))
    print("enter the filename")
    filename=str(input())
    #fig.show(renderer='vscode')
    fig.write_image("./../img/"+filename+".png")

    #plotly.offline.plot(fig, filename+'.html')

    return

if __name__ =="__main__":
    print("Run the script in skill-prediction directory")

    parser = argparse.ArgumentParser()
    parser.add_argument('--pathemb1', help='check path in data/dumps',default="./../../data/COLING/New-FT-true-label-concatCondensedData/tst_point_embs.npy")
    parser.add_argument('--pathemb2',help='check path in data/dumps',default="./../../data/COLING/New-FT-true-label-concatCondensedData/label_embs.npy") 
    parser.add_argument('--input',help='file that has all the corresponding labels to job',default="./../../data/COLING/tst_X_Y.txt") 
    parser.add_argument('--metric',help='cosine,euclidiean, or other',default="cosine")
    args = parser.parse_args()  
    if args.metric =="cosine-by-avg":
        print("Enter the value of k for top-k documents")
        k=int(input())
        cosine_sim_matrix=avg_calculate_cosine(args.pathemb1,args.pathemb2,args.input)
        cosine_sim_ind,cosine_sim_values=topkvalues(cosine_sim_matrix,k)
        visualize(cosine_sim_values)
    elif args.metric=="cosine":
        print("Enter the value of k for top-k documents")
        k=int(input())
        cosine_sim_matrix=calculate_cosine(args.pathemb1,args.pathemb2,args.input)
        print("cosine similarity matrix shape",len(cosine_sim_matrix))
        cosine_sim_ind,cosine_sim_values=topkvalues(cosine_sim_matrix,k)
        visualize(cosine_sim_values)
    else:
        print("nothing done")





