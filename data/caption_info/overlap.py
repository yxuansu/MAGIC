
cocoval_concepts = []
flikcr30kval_concepts = []

mode = "val"
with open(f"mscoco{mode}-concepts.txt", "r") as fin:
    for line in fin:
        cocoval_concepts.append(line.strip())

with open(f"flickr30k{mode}-concepts.txt", "r") as fin:
    for line in fin:
        flikcr30kval_concepts.append(line.strip())

cocoval_concepts = set(cocoval_concepts)
flikcr30kval_concepts = set(flikcr30kval_concepts)
print("coco => flickr:", len(cocoval_concepts & flikcr30kval_concepts) / len(flikcr30kval_concepts))
print("flickr => coco:", len(cocoval_concepts & flikcr30kval_concepts) / len(cocoval_concepts))