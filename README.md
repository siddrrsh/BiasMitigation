# Bias Mitigation Project

Siddharth, Rishi \

In this repo, we provide a full pipeline to train, combine, and analyze various bias mitigation methods. We also provide several metrics for comparison: \

## Metrics
Word Embedding Association Test (WEAT) - Caliskan et al. (2017) \
Relative Norm Distance (RND) - Garg et al. (2018), which has two variations (cosine and euclidean) \
Mean Average Cosine (MAC) - Manzini et al. (2019) \

## Usage
` python3 data-loader.py ` \
` debias.ipynb ` \
` pythonw evalbias.py ` \

## Requirements:
- Gensim, NumPy, SciPy