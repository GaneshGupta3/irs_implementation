import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from bs4 import BeautifulSoup
from itertools import combinations
import operator

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Instantiate NLP helpers
splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# --- Load datasets (assumes files are in same folder) ---
_df_norm = pd.read_csv('dis_sym_dataset_norm.csv')
df_norm = _df_norm.copy()

_df_comb = pd.read_csv('dis_sym_dataset_comb.csv')

# Precompute structures used in original notebook
columns_name = list(df_norm.columns)[1:]
N = len(df_norm)
M = len(columns_name)

# document (disease) names in order
documentname_list = list(df_norm['label_dis'])

# compute idf for each symptom
idf = {}
for col in columns_name:
    temp = np.count_nonzero(df_norm[col])
    # avoid division by zero
    idf[col] = float(np.log((N+1)/(temp+1)))

# tf (disease,symptom) and tf_idf
# tf stored in df_norm already (0/1 or counts)
# build tf_idf numpy matrix D (N x M)
D = np.zeros((N, M), dtype='float32')
for i, disease in enumerate(documentname_list):
    for j, col in enumerate(columns_name):
        tf = float(df_norm.iloc[i, j+1])
        D[i, j] = tf * idf.get(col, 0.0)

# cosine helper
def cosine_dot(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# tokenize + preprocess helper
def preprocess_text(text):
    text = text.lower()
    tokens = splitter.tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# gen_vector: builds TF-IDF query vector from tokens
def gen_vector(tokens):
    Q = np.zeros(M, dtype='float32')
    counter = Counter(tokens)
    for token in np.unique(tokens):
        tf = counter[token]
        try:
            idf_temp = idf[token]
        except KeyError:
            idf_temp = None
        try:
            ind = columns_name.index(token)
            if idf_temp is None:
                idf_temp = idf.get(columns_name[ind], 0.0)
            Q[ind] = tf * idf_temp
        except ValueError:
            # token not in columns_name
            continue
    return Q

# tf_idf_score function: sum tf_idf for matching query symptoms
def tf_idf_score(k, query):
    query_weights = {}
    for (disease, col), value in []:
        pass
    # original notebook used tf_idf dict; we can compute disease score by summing D[disease_idx, symptom_idx] for matched symptoms
    for idx, disease in enumerate(documentname_list):
        s = 0.0
        for sym in query:
            if sym in columns_name:
                j = columns_name.index(sym)
                s += float(D[idx, j])
        if s > 0:
            query_weights[disease] = s
    sorted_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    return sorted_weights[:k]

# cosine_similarity top-k
def cosine_similarity(k, query):
    qvec = gen_vector(query)
    d_cosines = [cosine_dot(qvec, D[i]) for i in range(N)]
    idxs = np.array(d_cosines).argsort()[-k:][::-1]
    out = {int(i): float(d_cosines[i]) for i in idxs if d_cosines[i] > 0}
    return out

# synonyms function (uses thesaurus.com + wordnet). It attempts to be resilient to failures.
def synonyms(term):
    syns = set()
    try:
        response = requests.get(f'https://www.thesaurus.com/browse/{term}', timeout=3)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            try:
                container = soup.find('section', {'class': 'MainContentContainer'})
                row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
                if row:
                    items = row.find_all('li')
                    for x in items:
                        syns.add(x.get_text().strip())
            except Exception:
                pass
    except Exception:
        # network or scraping failed; fallback to wordnet
        pass

    # WordNet lemmas
    for syn in wordnet.synsets(term):
        for lemma in syn.lemma_names():
            syns.add(lemma.replace('_', ' '))
    return syns

# Main prediction function (public)
def predict_disease_from_text(symptom_text: str, top_k: int = 10):
    """
    Input: single string containing symptoms (e.g. "fever, cough and headache")
    Output: dict with two lists: 'tfidf' and 'cosine', each is list of {disease, score}
    """
    # 1) preprocess input
    # allow separators by comma or 'and' etc.
    parts = [p.strip() for p in re_split_symptoms(symptom_text)]

    processed_user_symptoms = []
    for sym in parts:
        s = sym.replace('-', ' ').replace("'", '')
        s = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(s)])
        if s:
            processed_user_symptoms.append(s)

    # 2) query expansion: for each symptom, get synonyms for subsets
    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym_words = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym_words) + 1):
            for subset in combinations(user_sym_words, comb):
                subset = ' '.join(subset)
                str_sym.update(synonyms(subset))
        str_sym.add(' '.join(user_sym_words))
        # normalize underscores
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))

    # 3) match expanded user_symptoms to dataset symptom names
    found_symptoms = set()
    for data_sym in columns_name:
        data_sym_split = data_sym.split()
        for user_sym in user_symptoms:
            count = 0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count += 1
            if len(data_sym_split) > 0 and count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)
    found_symptoms = list(found_symptoms)

    # If nothing matched, try to use tokens directly
    if len(found_symptoms) == 0:
        # token-level match
        tokens = []
        for p in processed_user_symptoms:
            tokens += p.split()
        found_symptoms = [tok for tok in tokens if tok in columns_name]

    # 4) Build final_symp by choosing all matched symptoms (in CLI you asked user to select; here we take them all)
    final_symp = found_symptoms

    # 5) compute top-k using tf-idf sum and cosine similarity
    topk1 = tf_idf_score(top_k, final_symp)
    topk2 = cosine_similarity(top_k, final_symp)

    # prepare outputs as lists of dicts
    tfidf_out = [{'disease': name, 'score': round(float(score), 4)} for name, score in topk1]
    cosine_sorted = sorted(topk2.items(), key=lambda kv: kv[1], reverse=True)
    cosine_out = [{'disease': documentname_list[int(idx)], 'score': round(float(score), 4)} for idx, score in cosine_sorted]

    return {'tfidf': tfidf_out, 'cosine': cosine_out}

# small helper used above
import re
def re_split_symptoms(text):
    # split on commas or ' and ' or ' or '
    parts = re.split(r'[;,]|\band\b|\bor\b', text)
    return [p.strip() for p in parts if p.strip()]
