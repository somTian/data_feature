import json

with open('data/20170425.txt',encoding='utf-8') as f:
    for line in f:
        print(json.loads(line))