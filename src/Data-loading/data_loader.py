import subprocess

import utils
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from subprocess import Popen
from utils import load_json_pairs
from CDA_utils.substitutor import Substitutor
import numpy
import re
import os

LINK_REG = re.compile(r'\[ ([^[\]()]*) \] \( ([^[\]()]*) \)')
URL_REG = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def run(runfile):
  with open(runfile,"r") as rnf:
	  exec(rnf.read())

def main():
	print("Bias Analysis Program")
	corpus_src = "reddit.US.txt.tok.clean.shf.500K.nometa.tc.noent.fw.url"
	embeddings_src = ""
	val = input("Would you like to load pre-trained embeddings or train from a corpus (Enter 'p' for pre-trained or 't' for train from scratch): ")
	if val == "p":
		val2 = input("Would you like to use pre-trained embeddings with CDA? ('y' or 'n'): ")
		if val2 == "y":
			corpus_src = "pretrained_embeddings/CDA_embeddings.w2v"
		else:
			corpus_src = "pretrained_embeddings/embeddings.w2v"
		print("Your selected embedding file is: " + corpus_src)
	elif val == "t":
		CDA_input = input("Would you like to apply CDA to the Corpus ('y' or 'n'): ")
		if CDA_input == "y":
			print("Proceeding with CDA")
			print("Using Reddit L2 Corpus Data (U.S.A.) to generate Word Vectors ...")
			word2vec(True)
			embeddings_src = "CDA_corpus/CDA_reddit.US.cleanedforw2v.w2v"
		else:
			print("Proceeding without CDA")
			print("Using Reddit L2 Corpus Data (U.S.A.) to generate Word Vectors ...")
			word2vec(False)
			embeddings_src = "corpus/reddit.US.txt.tok.clean.shf.500K.nometa.tc.noent.fw.url.lc.cleanedforw2v.w2v"




def write_CDA_corpus(infile, outfile, newline='\n'):
	base_pairs = load_json_pairs('CDA-utils/cda_default_pairs.json')
	name_pairs = load_json_pairs('CDA-utils/names_pairs_1000_scaled.json')
	substitutor = Substitutor(base_pairs, name_pairs=name_pairs)

	f = open(infile, 'r')
	g = open(outfile, 'w')
	i = 0

	for line in f:
		if i+1 % 100000 == 0:
			print(i)
		match = URL_REG.match(line.strip()) 
		if match is None:
			text = preprocess(line)
		else:
			user, sub, text = match.group(1, 2, 3)
			text = preprocess(text)

		flipped = substitutor.invert_document(text) 
		if text != flipped: # no CDA was applied to this sentence/text (write just flipped) 
			g.write(flipped.strip().lower()) 
		else: # write original + flipped 
			g.write(text.strip().lower()) 
			g.write(flipped.strip().lower()) 
		i += 1




def isValidWord(word):
	return all([c.isalpha() for c in word])

def preprocess(text):
	# replace links of form: [ text ] ( url ) with just text
	text = LINK_REG.sub(r'\1', text)

	# replace urls with URL
	text = URL_REG.sub(r'<URL>', text)

	return text

def write_corpus(infile, outfile, newline='\n'):
	f = open(infile, 'r')
	g = open(outfile, 'w')
	i = 0

	for line in f:
		if i+1 % 100000 == 0:
			print(i)

		match = URL_REG.match(line.strip()) # no DATA_REG?
		if match is None:
			text = preprocess(line)
		else:
			user, sub, text = match.group(1, 2, 3)
			text = preprocess(text)

		g.write(text.strip().lower())
		g.write(newline)
		i += 1

def word2vec(CDA):
	for file in os.listdir("reddit.l2"):

		FILE = 'reddit.l2/' + file
		if CDA == True:
			OUTFILE = 'CDA_Corpus/' + file + ".cleanedforw2v"
		else:
			OUTFILE = 'corpus/' + file + ".cleanedforw2v"
		W2V_OUTFILE = OUTFILE + ".w2v"
		
		if (not os.path.isfile(W2V_OUTFILE) and not os.path.isfile(OUTFILE)):

			print("Constructing word vectors for " + str(file))
			print("Parsing Training Data")

			if CDA == True:
				print("Applying Counter Factual Data Augmentation on " + str(file))
				write_CDA_corpus(FILE, OUTFILE, newline="\n")
			else:
				write_corpus(FILE, OUTFILE, newline="\n")
			sentences = LineSentence(OUTFILE)

			print("Training w2v model... This may take a while")
			model = Word2Vec(sentences=sentences, vector_size=50, max_final_vocab=50000, workers=10, epochs=3)

			print("Training Complete... Got", len(model.wv.key_to_index), "unique word vectors")

			print("Saving vectors to disk")
			f = open(W2V_OUTFILE, "w")
			for word in model.wv.key_to_index:
				if(isValidWord(word)):
					try:
						f.write(word + " " + " ".join([str(float(x)) for x in model.wv[word]]) + "\n")
					except UnicodeEncodeError as e:
						pass
			f.close()
		else:
			print("Passing on", FILE, "because we found corresponding cleaned and w2v files")


def getPretrained():
	folder = './data/'


main()