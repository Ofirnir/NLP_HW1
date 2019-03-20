import numpy as np
import  re
from collections import Counter, OrderedDict, defaultdict
# n_gram=3


def train(text, n_gram):
    counter = Counter() #fa
    lines = text.splitlines()
    counter = Counter()

    for line in lines :
        line =['<s>', '<s>'] + list(line) + ['<e>','<e>']
        for i in range(0,len(line) - n_gram + 1 ):
         #   lin =  +
            t= line[i:i + n_gram]
            t2= tuple(t)
            counter[t2] += 1

    return counter


def lm(corpus_file, model_file):
    with open(corpus_file,'r',encoding="utf8") as file:
        data = file.read()
        unigram = train(data, 1)
        bigram = train(data, 2)
        trigram = train(data, 3)
        print(unigram)
        #print(bigram)
        # print(trigram)
        with open(model_file, 'w', encoding="utf8") as outputFile:
            print("3-grams:",file=outputFile,flush=False)
            for j in trigram:
                chars = bigram[j[0:2]]
                prob = np.log2(trigram[j] / (chars + 1) )
                #print("chars=",j[0: 2],chars , "i=",j,trigram[j], "prob=",prob)
                print(j, "\t",format("%.5f") % prob, file=outputFile,flush=False,sep="")

            print("\n2-grams:",file=outputFile,flush=False)
            for i in bigram:
                chars = unigram[i[0]]
                prob = np.log2(bigram[i] / (chars + 1) )
                # print("chars=",i[0: 2],chars , "i=",i,trigram[i], "prob=",prob)
                print(i, "\t", format("%.5f") % prob, file=outputFile, flush=False,sep="")
            print("\n1-grams:",file=outputFile,flush=False)

            for i in unigram:
                chars = unigram[i]
                prob = np.log2(chars / len(data))
                # print("chars=",i[0: 2],chars , "i=",i,trigram[i], "prob=",prob)
                print(i, "\t", format("%.5f") % prob, file=outputFile, flush=False,sep="")

def parse(model_file):
    with open(model_file, 'r', encoding="utf8") as modelf:
        data = modelf.readlines()
        parsingDict=defaultdict()
        for l in data:
            m = re.search("^(.*) \t (.*)",l)
            if m != None:
                parsingDict[m.group(1)]=float(m.group(2))
    return parsingDict

def eval(input_file,model_file,weights):
    with open(input_file,'r',encoding="utf8") as inputf:
        data=inputf.read()
        trigram=train(data,3)
        dict= parse(model_file)
        sum = 0
        for tri in trigram:
            if tri not in dict:
                sum += 0* 1/len(trigram)
            else :
                interpolate = weights[0]*dict[tri] + weights[1] * dict[tri[0:2]] + weights[2] * dict[tri[0]]
                sum+=interpolate
                #print( tri ,interpolate)
        sum/=-len(trigram)
        prex=np.power(2,sum)
        return prex
cor = "fr.csv"
model = 'model_file.txt'
languages =["en", "es", "fr", "in", "it","nl","pt", "tl"]
train_file="train"
test_file="test"
#shuffling and split to train/test
for lang in languages:
    cor=lang+".csv"
    with open(cor,'r',encoding="utf8") as lang_file:
        lines=lang_file.readlines()
        np.random.shuffle(lines)
        train_file=lang+"_train.txt"
        test_file=lang+"_test.txt"
        corpus_file=open(train_file,'w+',encoding="utf8")
        model_file=open(test_file,'w+',encoding="utf8")
        for i in range(0,int(0.9*len(lines)) ):
            corpus_file.write(lines[i])
        for i in range(int(0.9*len(lines)),len(lines)) :
            model_file.write(lines[i])
        corpus_file.close()
        model_file.close()

    #lm(train_file,model)
    #eval(test_file,model, [0.4, 0.3, 0.3])

#building lm for each language and computing perplexity

for lang in languages:
    train_file=lang+"_train.txt"
    lm(train_file,model)
    for test_lang in languages:
        test_file = lang + "_test.txt"
        prex=eval(test_file,model,[0.4, 0.3, 0.3])
        #fd d
        print("the eval of",test_lang,"on",lang,"==",prex)
