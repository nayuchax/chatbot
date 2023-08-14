import numpy as np
import pandas as pd
from pymagnitude import Magnitude
import re


# 前処理
def preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[0-9]", "0", text)
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)
    text = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)
    text = re.sub(
        r"[!”#$%&\’\\\\()*+,-./:;?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。,？！｀＋￥％]", "", text
    )

    return text


# chiveの読み込み(Magnitude形式)
def load_word2vec():
    model_dir = "data/chive-1.2-mc5.magnitude"
    model = Magnitude(model_dir)
    return model


# テキスト->単語に分割する
def tokenize(sent: str, tokenizer):
    morphs = tokenizer.tokenize(sent.strip())
    norms = []
    # 形態素の正規化形を取得
    for m in morphs:
        norms.append(m.normalized_form())
    return norms


# 名詞抽出
def noun_extraction(text: str, tokenizer):
    out = []
    content = [
        (m.surface(), m.dictionary_form(), m.reading_form(), m.part_of_speech())
        for m in tokenizer.tokenize(text)
    ]

    len_ = len(content)
    for i in range(len_):
        if content[i][3][0] == "名詞":
            out.append(content[i][0])

    return out


# テキスト->ベクトルに変換
def text2vec(text: str, vectorizer, tokenizer):
    norms = tokenize(text, tokenizer)
    word_vecs = []
    # ベクトルに変換
    for i in norms:
        if i in vectorizer:
            word_vecs.append(vectorizer.query([i])[0].flatten())
    # 平均ベクトルを計算
    text_vec = np.mean(word_vecs, axis=0)
    return text_vec


# csv読み込み
def load_csv():
    file_path = "data/test.csv"
    df = pd.read_csv(file_path, encoding="utf-8")
    return df


# 文書分類
def sentence_similarity(text1, text2, model, tokenizer):
    text1_avg_vector = text2vec(text1, model, tokenizer)
    text2_avg_vector = text2vec(text2, model, tokenizer)
    # コサイン類似度計算
    score = np.dot(text1_avg_vector, text2_avg_vector) / (
        np.linalg.norm(text1_avg_vector) * np.linalg.norm(text2_avg_vector)
    )
    return score
