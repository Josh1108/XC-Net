
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict as ddict
from gensim import models
import operator
import pandas as pd
import re
from xclib.data import data_utils
from gensim import corpora
from nltk.corpus import stopwords
from pos_tagging import pos_tag_job,remove_verb
from flashtext import KeywordProcessor

def word_tokenizer(text):
    return text.split()

def filter_regex(corpus):
    """
    Regex - single character and number and number sequences are removed.
    """
    stop_words = set(stopwords.words('english'))
    for i in range(len(corpus)):
        corpus[i] = re.sub(r'[0-9]+', '', corpus[i])
        corpus[i] =  re.sub(r"\b[a-zA-Z]\b", "", corpus[i])
        word_tokens = corpus[i].split()
        filt_sent =[w for w in word_tokens if not w.lower() in stop_words]
        corpus[i] = " ".join(filt_sent)
    return corpus
    
def preprocess(corpus,type:str = 'tf_idf',k=10):
    corpus = filter_regex(corpus)
    if type=='tf-idf':
        return tf_idf(corpus,k)
    elif type =='freq':
        return frequency_preprocess(corpus,k)
    elif type == 'label-sent':
        return true_label_preprocess(corpus)
    elif type =='true-label-concat':
        return true_label_concat(corpus)
    elif type =='lbl-from-job':
        return lbl_from_job(corpus)
    elif type=='tf-idf-gensim':
        return tf_idf_gensim(corpus,k)
    elif type =='pos-tag':
        return pos_tag(corpus)
    else:    
        return NotImplementedError

def tf_idf_gensim(corpus,k=10):
    """
    remove the k most unimportant words in a job based on tf-idf scores
    """
    processed_corpus =[]

    # remove words that appear only once
    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1 
    #         texts = [[token for token in text if frequency[token] > 1]for text in texts]
    documents =[x.split() for x in corpus]

    dictionary = corpora.Dictionary(documents)
    _corpus = [dictionary.doc2bow(text) for text in documents]
    tfidf = models.TfidfModel(_corpus,normalize=True)
    corpus_tfidf = tfidf[_corpus]

    for i,doc in tqdm(enumerate(corpus_tfidf)):

        doc.sort(key = lambda x: x[1])

        lis_items = []
        for j in range(k):
            if j<len(doc):
                    lis_items.append(dictionary[doc[j][0]])
        pre_item =[]
        for item in documents[i]:
            if item not in lis_items:
                pre_item.append(item)
        str_pre_item = " ".join(pre_item)
        if str_pre_item!="":
            processed_corpus.append(str_pre_item)
        else:
            processed_corpus.append(corpus[i])
    return processed_corpus

def pos_tag(corpus):
    corpus_filt = []
    for corp in tqdm(corpus):
        corp_filt = remove_verb(corp)
        corpus_filt.append(corp_filt)
    assert len(corpus) == len(corpus_filt)
    return corpus_filt

def tf_idf(corpus,k=10):
    """
    remove the k most unimportant words in a job based on tf-idf scores
    """
    vectorizer = TfidfVectorizer(tokenizer=word_tokenizer)
    X = vectorizer.fit_transform(corpus)
    Y = vectorizer.get_feature_names()
    full_corpus_new = []

    for i in tqdm(range(len(corpus))):
        dicti = dict(zip(vectorizer.get_feature_names(), X.toarray()[i]))
        lisi=[]
        for text in corpus[i].split():
            lisi.append(dicti[text])
        lisi_index = sorted(range(len(lisi)), key=lambda k: lisi[k])
        lisi_index=lisi_index[:k]
        corpus_split = corpus[i].split()
        corpus_split_pre = []

        for i, item in enumerate(corpus_split):
            if i in lisi_index:
                continue
            else:
                corpus_split_pre.append(item)
        corpus_preprocessed = " ".join(corpus_split_pre)
        full_corpus_new.append(corpus_preprocessed)
    # print(full_corpus_new[:5])
    return full_corpus_new
def tf_idf_labels(corpus):
    """
    Keep only labels after tf-idf
    """
    courpus = tf_idf_gensim(corpus)
    corpus = lbl_from_job(corpus)
    return corpus
def frequency_preprocess(corpus,k=10):
    """
    Globally check the top 'k' frequent words and then delete them from
    the entire corpus
    """
    count_dict =ddict(int)
    keys_to_remove=[]
    filtered_corpus =[]

    for item in corpus:
        for word in item.split():
            count_dict[word]+=1

    sorted_x = sorted(count_dict.items(), key=operator.itemgetter(1))
    
    for item in sorted_x[:k]:
        key,_ = item
        keys_to_remove.append(key)

    for item in tqdm(corpus):
        querywords = item.split()
        resultwords  = [word for word in querywords if word.lower() not in keys_to_remove]
        result = ' '.join(resultwords)
        filtered_corpus.append(result)

    return filtered_corpus

def generate_df(data_frame_path:str,filtered_corpus,column_name:str)->pd.DataFrame:
    
    df = pd.read_csv(data_frame_path)
    df[column_name] = filtered_corpus
    df.to_csv(data_frame_path, index=None)
    return df

def true_label_preprocess(corpus,data=None):
    """
    Keep those sentences where true labels occur
    NOT CORRECT.
    """
    if data == None:
        data = data_utils.read_sparse_file('/home/jsk/skill-prediction/data/COLING/trn_X_Y.txt',force_header =True).toarray()
        print("loaded test file true labels from absolute path")

    f = open("/home/jsk/skill-prediction/data/COLING/Y.txt")
    labels=f.readlines()
    labels = [x.split('\n')[0] for x in labels]
    corpus_filtered =[]

    for i, item in enumerate(corpus):
        
        ind = [j for j,x in enumerate(data[i]) if x==1]
        labels_i=[labels[x] for x in ind]

        jd =""
        for text in item.split():
            print("Text printed",text)
            Flag =0
            for lbl in labels_i:
                # print("label",lbl)
                if lbl in text:
                    Flag =1
            if Flag ==1:
                print("label added",text)
                jd+=text
        if jd =="":
            print("empty jd")
            jd = item
        corpus_filtered.append(jd)
    return corpus_filtered

def lbl_from_job(corpus):
    """
    Keep only labels. Label dictionary compared against JD and intersecting words are kept.
    """
    f = open("/home/jsk/skill-prediction/data/COLING/Y.txt")
    labels=f.readlines()
    labels = [x.split('\n')[0] for x in labels]
    corpus_filtered =[]
    keyword_processor = KeywordProcessor()
    for x in labels:
        keyword_processor.add_keyword(x)

    for i, item in enumerate(corpus):
        jd =""
        keywords_found = keyword_processor.extract_keywords(item)
        keywords_found =[x.lower() for x in keywords_found]
        
        jd =" ".join(keywords_found)
        
        if jd =="":
            jd = item
        corpus_filtered.append(jd)
    return corpus_filtered
def true_label_concat(corpus=None,data=None):
    """
    Concatonation of ground truth labels. use corpus only for indexing.
    """
    if data == None:
        data = data_utils.read_sparse_file('/home/jsk/skill-prediction/data/COLING/trn_X_Y.txt',force_header =True).toarray()
        print("loaded train file true labels from absolute path -- make sure it's not test")

    f = open("/home/jsk/skill-prediction/data/COLING/Y.txt")
    labels=f.readlines()
    labels = [x.split('\n')[0] for x in labels]
    corpus_filtered =[]

    for i, item in enumerate(corpus):
        ind = [j for j,x in enumerate(data[i]) if x==1]
        labels_i=[labels[x].lower() for x in ind]
        jd = " ".join(labels_i)
        corpus_filtered.append(jd)
    assert len(corpus_filtered) == len(corpus)
    print(corpus_filtered[0])
    return corpus_filtered


# preprocess_type = 'true-lbl-from-job'
# path = '/home/jsk/skill-prediction/data/COLING/trn_X.txt'
# file=open(path,"r")
# jobs=file.readlines()
# jobs = [x.split('\n')[0] for x in jobs]
# jobs = preprocess(jobs[0:5],preprocess_type)
# # print(jobs[0])