# 形態複素解析 mecab
# 辞書 unidec lite

import re
import MeCab

def preprocess(text: str) -> str:
    #改行コードの削除
    text = text.replace("\n", "")
    text = text.replace("\r", "")

    # 小文字に統一
    text = text.lower()

    return text

def morphological_analysis(text: str) -> list[str | int]:
    tagger = MeCab.Tagger("-Owakati")
    divide_text = tagger.parse(text).split()

    return divide_text

text = "すもももももももものうち."

print(morphological_analysis(preprocess(text)))