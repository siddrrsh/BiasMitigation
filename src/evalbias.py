'''
System Requirements: Responsibly (pip install responsibly), Gensim (pip install gensim)
Must run using 'pythonw' not 'python3/python' (Responsibly.ai requirement as part of Anaconda)


Uses Responsibly.ai:
Bolukbasi et al. (2016) bias measure and debiasing - responsibly.we.bias
WEAT measure - responsibly.we.weat
Gonen et al. (2019) clustering as classification of biased neutral words - responsibly.we.bias.BiasWordEmbedding.plot_most_biased_clustering()

Raw Approach:
Word Embedding Association Test (WEAT) - Caliskan et al. (2017)
Relative Norm Distance (RND) - Garg et al. (2018), which has two variations (cosine and euclidean)
Mean Average Cosine (MAC) - Manzini et al. (2019)
'''

import subprocess
from responsibly.we import GenderBiasWE
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim import downloader
#from wefe import WordEmbeddingModel, Query, WEAT
#from wefe.utils import run_queries
#from wefe.datasets import load_weat
import gensim.downloader as api
import numpy as np
import gensim
import numpy
import re
import os


def run(runfile):
  with open(runfile,"r") as rnf:
	  exec(rnf.read())

def main():
	print("Bias Evaluation Program: This program performs evaluation on existing word embeddings.")
	embeddings_src = "pretrained_embeddings/"
	val = input("Please specify the embeddings you wish to use by inputting an integer based on the following list of options: 1 -> 'Bolukbasi with CDA', 2 -> 'Bolukbasi with no CDA', 3 -> 'pretrained with CDA', 4 -> 'pretrained without CDA': ")
	val = int(val)
	if val == 1:
		embeddings_src += "Bolukbasi_CDA.w2v"
	elif val == 2:
		embeddings_src += "Bolukbasi_noCDA.w2v"
	elif val == 3:
		embeddings_src += "CDAembeddings.w2v"
	elif val == 4:
		embeddings_src += "embeddings.w2v"
	else:
		print("You selected an invalid option. Try again.")
	E = embeddings_src
	getMetrics(E)



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


def getMetrics(E):
	E = "pretrained_embeddings/modified_embeddings.w2v"
	f = open(E, "r")
	data = f.read()
	words = data.split()
	dim = 50
	size = len(words)/(dim+1) 
	line_to_add = str(int(size)) + " " + str(dim) # length + dimension
	prepend_line(E, line_to_add)

	model = KeyedVectors.load_word2vec_format(E, binary=False)
	w2v_gender_bias_we = GenderBiasWE(model)
	print(w2v_gender_bias_we.calc_direct_bias())


'''
def WEAT(E):
	word2vec_model = WordEmbeddingModel(E)

	# target sets (sets of popular names in the US)
	male_names = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
	female_names = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']

	#attribute sets
	career = ['executive', 'management', 'professional', 'corporation',
         'salary', 'office', 'business', 'career']
	family = ['home', 'parents', 'children', 'family', 'cousins', 'marriage',
         'wedding', 'relatives']

	gender_occupation_query = Query([male_names, female_names],
                                [career, family],
                                ['Male names', 'Female names'],
                                ['Career', 'Family'])
	weat = WEAT()
	results = weat.run_query(gender_occupation_query, word2vec_model)
	# Load the sets used in the weat case study
	weat_wordsets = load_weat()

	gender_math_arts_query = Query(
    	[male_names, female_names],
    	[weat_wordsets['math'], weat_wordsets['arts']],
    	['Male names', 'Female names'],
    	['Math', 'Arts'])

	gender_science_arts_query = Query(
    	[male_names, female_names],
    	[weat_wordsets['science'], weat_wordsets['arts_2']],
    	['Male names', 'Female names'],
    	['Science', 'Arts'])
'''


main()


