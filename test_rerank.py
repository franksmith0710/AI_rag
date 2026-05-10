import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"D:\hf_models\BAAI\bge-reranker-v2-m3"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

query = "怎么在Python里安装依赖包？"
documents = [
    "Python的pip命令可以用来安装第三方库，比如pip install requests",
    "红烧肉的做法：五花肉焯水，加冰糖、生抽、老抽慢炖",
    "Python虚拟环境可以隔离项目依赖，避免版本冲突",
    "Python的FlagEmbedding库可以用来加载bge系列重排模型"
]

pairs = [[query, doc] for doc in documents]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))

with torch.no_grad():
    outputs = model(**inputs)
    raw_scores = outputs.logits.squeeze(-1).tolist()

scores = [sigmoid(s) for s in raw_scores]

print("问题：", query)
print("\n文档打分结果：")
for doc, score in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True):
    print(f"分数：{score:.4f} | 文档：{doc}")
