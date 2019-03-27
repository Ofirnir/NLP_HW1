import numpy as np
import re
from collections import Counter, defaultdict
import pandas as pd


def train(text, n_gram):
    """
    Saves the frequency of each n_gram in the text file, text.

    :param text: plain text file.
    :param n_gram: the number to divide the chars by.
    :return: the counter of frequencies of each n_gram.
    """
    lines = text.splitlines()
    counter = Counter()

    for line in lines:
        line = ['<s>', '<s>'] + list(line) + ['<e>', '<e>']
        for i in range(0, len(line) - n_gram + 1):
            t = line[i:i + n_gram]
            t2 = tuple(t)
            counter[t2] += 1

    return counter


def lm(corpus_file, model_file):
    """
    Generates a language model over characters from a textual corpus.
    The uni-gram model is smoothed with Add One (Laplace), so it also has the <unk> probability for unknown characters.

    :param corpus_file: plain text file path.
    :param model_file: the generated file. There are 3 blocks in the file, separated by a space line. The first block
    have all the 3 letter sequences for the tri-gram model, as extracted from the input corpus, along with their log
    probabilities. Similarly the second block has all the bi-grams, and then all the uni-grams.
    :return: void
    """
    with open(corpus_file, 'r', encoding="utf8") as file:
        data = file.read()
        unigram = train(data, 1)
        bigram = train(data, 2)
        trigram = train(data, 3)
        with open(model_file, 'w', encoding="utf8") as outputFile:
            print("3-grams:", file=outputFile)
            for j in trigram:
                chars = bigram[j[0:2]]
                prob = np.log2(trigram[j] / chars)
                print(j[0], j[1], j[2], "\t", format("%.5f") % prob, file=outputFile, sep="")

            print("\n2-grams:", file=outputFile)
            for i in bigram:
                chars = unigram[(i[0],)]
                prob = np.log2(bigram[i] / chars)
                print(i[0], i[1], "\t", format("%.5f") % prob, file=outputFile, sep="")

            print("\n1-grams:", file=outputFile)
            prob = np.log2(1 / (len(data) + len(unigram)))
            print("<unk>\t", format("%.5f") % prob, file=outputFile, sep="")
            for k in unigram:
                chars = unigram[k]
                prob = np.log2((chars + 1) / (len(data) + len(unigram)))
                print(k[0], "\t", format("%.5f") % prob, file=outputFile, sep="")


def parse(model_file):
    """
    Parsing each n_gram and its probability into a dictionary.

    :param model_file: plain text file path.
    :return: a dictionary.
    """
    with open(model_file, 'r', encoding="utf8") as modelf:
        data = modelf.readlines()
        parsing_dict = defaultdict()
        for l in data:
            m = re.search('^(.*)\t(.*)', l)
            if m is not None:
                matches = (m.group(1))
                parsing_dict[matches] = float(m.group(2))
                # else:  parsingDict [matches[0]] = float(m.group(2))
    return parsing_dict


def eval(input_file, model_file, weights):
    """
    Calculates and prints the perplexity of a model running over a given text.
    Builds a trigram model, using interpolation with 3 weights for the three models.

    :param input_file: plain text file path.
    :param model_file: plain text file path.
    :param weights: the weights of each model: tri-gram, bi-gram and uni-gram respectively The sum of weights has to be one.
    :return: the perplexity.
    """
    with open(input_file,'r',encoding="utf8") as inputf:
        data = inputf.read().splitlines()
        # trigram=train(data,3)
        dict = parse(model_file)
        sum = 0
        size = 0
        for line in data:
            size += len(line)
            line = ['<s>', '<s>'] + list(line) + ['<e>', '<e>']
            for i in range(0, len(line)-2):
                string = line[i]+line[i+1] + line[i+2]
                if string in dict:
                   interpolate = weights[0] * dict[string] + weights[1] * dict[line[i] + line[i + 1]] + weights[2] * dict[line[i]]
                elif ((line[i] + line[i + 1]) in dict):
                    interpolate= weights[1] * dict[line[i] + line[i + 1]] + weights[2] * dict[line[i]]
                elif line[i] in dict :
                    interpolate= weights[2] * dict[line[i]]
                else: interpolate= dict["<unk>"]
                sum+= interpolate
        sum /= -size
        prex = pow(2, sum)
    return prex


def main():
    """
    Creates for each file a 90/10 train/dev random split and use the train part for building a model, using lm,
    then run eval on the testing file of every language.
    Creates a plain text file with a table of the results.

    :return: void
    """
    languages =["en", "es", "fr", "in", "it", "nl", "pt", "tl"]

    # shuffling and split to 90/10 train/test
    for lang in languages:
        cor = lang+".csv"
        with open(cor, 'r', encoding="utf8") as lang_file:
            lines = lang_file.readlines()
            np.random.shuffle(lines)
            train_file = lang+"_train.txt"
            test_file = lang+"_test.txt"
            corpus_file = open(train_file, 'w+', encoding="utf8")
            model_file = open(test_file, 'w+', encoding="utf8")
            for i in range(0, int(0.9*len(lines))):
                corpus_file.write(lines[i])
            for i in range(int(0.9*len(lines)), len(lines)):
                model_file.write(lines[i])
            corpus_file.close()
            model_file.close()

    # building lm for each language and computing perplexity
    model = 'model_file.txt'
    # building the model
    i = 0
    j = 0
    perplexity_data = np.empty([len(languages), len(languages)])
    for lang in languages:
        train_file = lang+"_train.txt"
        lm(train_file, model)
        for test_lang in languages:  # test every language
            test_file = test_lang + "_test.txt"
            prex = eval(test_file, model, [0.4, 0.3, 0.3])
            perplexity_data[i, j] = prex
            j += 1
        i += 1
        j = 0

    # the results formatted as a table
    # The value in cell (i, j) is the perplexity calculated by evaluating the model of the language in row i using the
    # test file of the language in column j.
    df = pd.DataFrame(data=perplexity_data, index=languages, columns=languages)
    df.to_csv('final_output.csv')


if __name__ == "__main__":
    main()
