import word2vec
import sudachipy

# ダウンロードしたchiVeを読み込み
model = word2vec.load_word2vec()
# トークナイザ
tokenizer = sudachipy.Dictionary().create()

out_data = []
inp = input("検索入力: ")
df = word2vec.load_csv()
q_data = list(df["q"])
a_data = list(df["a"])

for i, data in enumerate(df["q"].str.cat(df["a"])):
    try:
        score = word2vec.sentence_similarity(
            word2vec.preprocessing(inp), word2vec.preprocessing(data), model, tokenizer
        )
        out_data.append([i, score])
    except KeyError as error:
        print(error)

data = sorted(out_data, reverse=True, key=lambda x: x[1])

for i in range(3):
    if data[i][1] > 0.8:
        print("question : " + str(q_data[data[i][0]]))
        print("answer : " + str(a_data[data[i][0]]))
        print("score : " + str(data[i][1]))
