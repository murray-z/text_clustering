# -*- coding: UTF-8 -*-

import os
import sys
import jieba
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SinglePassCluster():
    def __init__(self, stopWords_path="../data/stop_words.txt", my_stopwords=None,
                 max_df=0.5, max_features=1000,
                 simi_threshold=0.5, res_save_path="./cluster_res.json"):
        self.stopwords = self.load_stopwords(stopWords_path)
        if isinstance(my_stopwords, list):
            self.stopwords += my_stopwords
        self.tfidf = TfidfVectorizer(stop_words=self.stopwords, max_df=max_df, max_features=max_features)
        self.simi_thr = simi_threshold
        self.cluster_center_vec = [] # [cluster_center_vec, ]
        self.idx_2_text = {}  # {文本id: text, }
        self.cluster_2_idx = {}  # {cluster_id: [text_id, ]}
        self.res_path = res_save_path  # save self.cluster_2_idx

    def load_stopwords(self, path):
        stopwords = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                stopwords.append(line.strip())
        return stopwords

    def cut_sentences(self, texts):
        if isinstance(texts, str):
            if not os.path.exists(texts):
                 print("path: {} is not exist !!!".format(texts))
                 sys.exit()
            else:
                _texts = []
                with open(texts, 'r', encoding="utf-8") as f:
                    for line in f:
                        _texts.append(line.strip())
                texts = _texts
        texts_cut = [" ".join(jieba.lcut(t)) for t in texts]
        self.idx_2_text = {idx: text for idx, text in enumerate(texts)}
        return texts_cut


    def get_tfidf(self, texts_cut):
        tfidf = self.tfidf.fit_transform(texts_cut)
        return tfidf.todense().tolist()

    def cosion_simi(self, vec):
        simi = cosine_similarity(np.array([vec]), np.array(self.cluster_center_vec))
        max_idx = np.argmax(simi, arixs=1)[0]
        max_val = simi[0][max_idx]
        return max_val, max_idx

    def single_pass(self, texts):
        texts_cut = self.cut_sentences(texts)
        tfidf = self.get_tfidf(texts_cut)
        # print(len(tfidf), len(tfidf[0]))

        # 开始遍历
        for idx, vec in enumerate(tfidf):
            # 初始化，没有中心生成
            if not self.cluster_center_vec:
                self.cluster_center_vec.append(vec)
                self.cluster_2_idx[0] = [idx]
            # 存在簇
            else:
                max_simi, max_idx = self.cosion_simi(vec)
                if max_simi >= self.simi_thr:
                    self.cluster_2_idx[max_idx].append(idx)
                else:
                    self.cluster_center_vec.append(vec)
                    self.cluster_2_idx[len(self.cluster_2_idx)] = [idx]

        with open(self.res_path, "w", encoding="utf-8") as f:
            json.dump(self.cluster_2_idx, f, ensure_ascii=False)



if __name__ == "__main__":
    test_data = "../data/test_data.txt"
    cluster = SinglePassCluster(max_features=100, simi_threshold=0.1)
    cluster.single_pass(test_data)





