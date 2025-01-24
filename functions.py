import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from nltk.stem import *
from nltk.stem.porter import *

import os
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))

csv_file_path = os.path.join(current_dir, 'corpus.csv')

df = pd.read_csv(csv_file_path)

import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

pd.set_option('display.max_colwidth', 150)

nltk.download('stopwords')
print(stopwords.words('english'))
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

stemmer = PorterStemmer()

def Steem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def clean(text):
   text = re.sub(r"http\S+", " ", text)
   text = re.sub(r"RT ", " ", text)
   text = re.sub(r"@[\w]*", " ", text)
   text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)
   text = re.sub(r'\t', ' ', text)
   text = re.sub(r'\n', ' ', text)
   text = re.sub(r"\s+", " ", text)
   text = text.strip()
   return text

def preprocess(sentence):
  sentence = remove_stopwords(sentence)
  sentence = clean(sentence)
  sentence = Steem_text(sentence)

  return sentence

df['processed_text'] = df['text'].apply(preprocess)

indexer = pt.DFIndexer("E:\ZC\YEAR 2 S2\DSAI 201\Project\myFirstIndex", overwrite=True)
index_ref = indexer.index(df["processed_text"], df["docno"])

index = pt.IndexFactory.of(index_ref)



current_dir = os.path.dirname(os.path.realpath(__file__))

csv_file_path = os.path.join(current_dir, 'queries.csv')

df_queries = pd.read_csv(csv_file_path)

df_queries = df_queries.rename(columns={'_id': 'qid'})

qid_mapping = {qid: i for i, qid in enumerate(df_queries['qid'].unique(), 1)}

df_queries['qid'] = df_queries['qid'].map(qid_mapping)

df_queries['query'] = df['text'].apply(preprocess)


retr = pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"})

retr.setControl("wmodel", "TF_IDF")
retr.setControls({"wmodel": "TF_IDF"})

df_queries['query'] = df_queries['query'].str.replace("'", "")
df_queries['query'] = df_queries['query'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

res = retr.transform(df_queries)
res

rm3_expander = pt.rewrite.RM3(index,fb_terms=10)
rm3_qe = retr >> rm3_expander
expanded_query = rm3_qe.transform(df_queries.iloc[0]["query"])

res = retr.transform(expanded_query)


def search_function(query):
  search_results = []

  q = preprocess(query)
  expanded_query = rm3_qe.search(q).iloc[0]["query"]
  expanded_query_formatted = ' '.join(expanded_query.split()[1:])
  expanded_query_formatted = [arr.split("^")[0] for arr in expanded_query.split()[1:]]
  expanded_query_formatted = " ".join(expanded_query_formatted)
  results_wqe = retr.search(expanded_query_formatted)

  for doc_id in results_wqe["docid"]:
    doc_text = df.iloc[int(doc_id)]["text"]
    search_results.append({"doc_id": doc_id, "text": doc_text})

  return search_results