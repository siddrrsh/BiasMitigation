from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import random
import numpy as np

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_professions


# load google news word2vec
E = WordEmbedding('Original_RedditUS_embeddings_cleaned.w2v')

# load professions
professions = load_professions()
profession_words = [p[0] for p in professions]

# gender direction
v_gender = E.diff('she', 'he')

# analogies gender
a_gender = E.best_analogies_dist_thresh(v_gender)

'''
for (a,b,c) in a_gender:
    print(a+"-"+b)
'''

# profession analysis gender
sp = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])

print(sp[0:20], sp[-20:])

# gender debiasing


from debiaswe.debias import debias

# Lets load some gender related word lists to help us with debiasing
with open('./data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_pairs.json', "r") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_seed.json', "r") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

debias(E, gender_specific_words, defs, equalize_pairs)

# profession analysis gender
sp_debiased = sorted([(E.v(w).dot(v_gender), w) for w in profession_words])

sp_debiased[0:20], sp_debiased[-20:]


# analogies gender
a_gender_debiased = E.best_analogies_dist_thresh(v_gender)

for (a,b,c) in a_gender_debiased:
    print(a+"-"+b)