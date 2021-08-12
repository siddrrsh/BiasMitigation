'''
System Requirements: Gensim (pip install gensim)
Must run using 'pythonw' not 'python3/python' (Gensim requirement as part of Anaconda)

Metrics we compute:
Word Embedding Association Test (WEAT) - Caliskan et al. (2017)
Relative Norm Distance (RND) - Garg et al. (2018), which has two variations (cosine and euclidean)
Mean Average Cosine (MAC) - Manzini et al. (2019)
'''

import subprocess
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim import downloader
import gensim.downloader as api
from word_list import target_words, aligned_word_lists, unaligned_word_lists
from aligned_estimators import compute_all_aligned_estimates
from unaligned_estimators import compute_all_unaligned_estimates
from WordEmbedding import WordEmbedding
import pandas as pd
import numpy as np
import numpy
import sys
import re
import os


def run(runfile):
  with open(runfile,"r") as rnf:
	  exec(rnf.read())

def main():
	print("Bias Evaluation Program: This program performs evaluation on existing word embeddings.")
	embeddings_src = "../Pretrained_Embeddings/"
	val = input("Please specify the embeddings you wish to use by inputting an integer based on the following list of options: 1 -> 'Bolukbasi with CDA', 2 -> 'Bolukbasi with no CDA', 3 -> 'pretrained with CDA', 4 -> 'pretrained without CDA', 5 -> 'compute for all files': ")
	val = int(val)
	if val == 1:
		embeddings_src += "Bolukbasi_CDA.w2v"
	elif val == 2:
		embeddings_src += "Bolukbasi_noCDA.w2v"
	elif val == 3:
		embeddings_src += "CDAembeddings.w2v"
	elif val == 4:
		embeddings_src += "embeddings.w2v"
	elif val == 5:
		outputs_dict = {}
		file_count = 0
		for filename in os.listdir(embeddings_src):
			if filename != ".DS_Store":
				print(filename)
				file = embeddings_src + filename
				w2v_dict = EmbeddingFileToDict(file)
				aligned_res = evalAlignedBias(w2v_dict)
				unaligned_res = evalUnalignedBias(w2v_dict)
				outputs_dict[file] = writeOutput(aligned_res, unaligned_res)
				file_count += 1
				if file_count == 8:
					break
		print(pd.DataFrame.from_dict(outputs_dict))
		quit()

	else:
		print("You selected an invalid option. Try again.")
	E = embeddings_src
	w2v_dict = EmbeddingFileToDict(E)
	aligned_res = evalAlignedBias(w2v_dict)
	unaligned_res = evalUnalignedBias(w2v_dict)
	writeOutput(aligned_res, unaligned_res)



def EmbeddingFileToDict(E):
	dict = {}
	file = open(E, "r")
	for line in file:
		values = line.split()
		curr_word = ""
		count = 0
		for value in values:
			if count == 0:
				curr_word = str(value)
				dict[curr_word] = []
			else:
				dict[curr_word].append(float(value))
			count += 1
		dict[curr_word] = np.array(dict[curr_word]) # convert to np array
	file.close()
	return dict



def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def convertToKeyedVectors(E):
	E = "../Pretrained_Embeddings/modified_embeddings.w2v"
	f = open(E, "r")
	data = f.read()
	words = data.split()
	dim = 50
	size = len(words)/(dim+1) 
	line_to_add = str(int(size)) + " " + str(dim) # length + dimension
	prepend_line(E, line_to_add)
	model = KeyedVectors.load_word2vec_format(E, binary=False)
	return model


# [embeddings] - Map from words to vectors
# [A1], [A2] - Lists of words for the two social groups of interest (e.g. male, female)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores

def evalAlignedBias(word_vectors):
	seed_pairs = aligned_word_lists
	professions = target_words['bolukbasi_professions']
	pooling = abs
	aligned_results_dict = compute_all_aligned_estimates(word_vectors, seed_pairs, professions, pooling, verbose=False)
	return aligned_results_dict


# [embeddings] - Map from words to vectors
# [A_list] - List of list of words. Each inner list designates a social group (e.g. white, black, asian)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores

def evalUnalignedBias(word_vectors):
	# no castillo, curator, firebrand, organist, confesses (from original word list)
	seed_pairs = aligned_word_lists
	all_religious_groups = [["gonzalez", "sanchez", "rivera", "martinez", "torres", "rodriguez", "perez", "lopez", "medina", "diaz", "garcia", "castro", "cruz"],
    ["cho", "wong", "tang", "huang", "chu", "chung", "ng", "wu", "liu", "chen", "lin", "yang", "kim", "chang", "shah", "wang", "li", "khan", "singh", "hong"],
    ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation", "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross", "catholic", "church"],
    ["allah", "ramadan", "turban", "emir", "salaam", "sunni", "koran", "imam", "sultan", "prophet", "veil", "ayatollah", "shiite", "mosque", "islam", "sheik", "muslim", "muhammad"],
    ["judaism", "jew", "synagogue", "torah", "rabbi"]]
	gender_groups = [["he", "son", "his", "him", "father", "man", "boy", "himself", "male", "brother", "sons", "fathers", "men", "boys", "males", "brothers", "uncle", "uncles", "nephew", "nephews"], ["she", "daughter", "her", "mother", "woman", "girl", "herself", "female", "sister", "daughters", "mothers", "women", "girls", "sisters", "aunt", "aunts", "niece", "nieces"]] 
	# removed 'hers'
	professions = target_words['garg_professions']
	pooling = abs
	unaligned_results_dict = compute_all_unaligned_estimates(word_vectors, gender_groups, professions, pooling, verbose=False)
	return unaligned_results_dict # gender bias results


def writeOutput(aligned, unaligned):
	results = {}
	for key in aligned:
		results[key] = aligned[key]
	for key in unaligned:
		results[key] = unaligned[key]
	return results
	




	




main()

