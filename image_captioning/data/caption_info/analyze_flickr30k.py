import json
from tqdm import tqdm

import nltk
from nltk.chunk import tree2conlltags

mode = "val"
with open(f"../datasets/flickr30k/flickr30k_{mode}.json", "r") as fin:
    data = json.load(fin)
print("Num of Images:", len(data))

all_anns = []
for da in data:
    all_anns += da['captions'] 

tokens = []
for sent in all_anns:
    tokens += sent.strip().split()
print("Num of unique words:", len(set(tokens)))

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def parse(sent):
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    return iob_tagged

concepts = []
for ann in tqdm(all_anns):
    sent = preprocess(ann)
    iob_tagged = parse(sent)
    concepts += [iob[0] for iob in iob_tagged if iob[2] == "I-NP"]
concepts = list(set(concepts))
print("Num of concepts:", len(concepts))

with open(f"flickr30k{mode}-concepts.txt", "w") as fout:
    for conc in concepts:
        fout.write(f"{conc}\n")