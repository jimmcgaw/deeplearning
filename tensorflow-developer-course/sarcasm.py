import json

import tensorflow as tf

sentences = []
is_sarcastic = []
with open("./Sarcasm_Headlines_Dataset.json", "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        sentences.append(
            data['headline']
        )
        is_sarcastic.append(
            data['is_sarcastic']
        )

print(len(sentences))
print(len(is_sarcastic))