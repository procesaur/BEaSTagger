import os
import sys
import pathlib
import stanza
import re
from os import path
from beast.StanzaTagger.stanzaworks import getScores


probability = False
lemmatize = True
tokenize = False

if probability:
    lemmatize = False


def tag_stanza(par_path, file_path, out_path, probability, lemmatize, tokenize):

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().rstrip()

    text = re.sub(r'\n\n+', '\n\n', text).strip()
    par = pathlib.PurePath(par_path).name
    pardir = par_path.rstrip(par + '/')
    sents = text.split("\n\n")
    tokens = []
    for s in sents:
        tokens.append(s.split("\n"))

    pt = par_path + "/../standard.pt"
    parx = os.listdir(par_path)[0]
    nlp = stanza.Pipeline(par, dir=pardir, processors='tokenize,pos', tokenize_pretokenized=True,
                          pos_model_path=par_path + "/" + parx, pos_pretrain_path=pt, logging_level='FATAL')

    document = nlp(tokens)
    scores, preds, newdoc = getScores(nlp, document)

    labels = nlp.processors["pos"].trainer.vocab['upos']

    original = sys.stdout
    sys.stdout = open(out_path, 'w', encoding='utf-8')

    si = 0
    for sidx, sent in enumerate(newdoc.sentences):
        for tidx, token in enumerate(sent.tokens):
            tokprob = ""
            lemma = ""
            if token.text != " ":
                tokprob = ''
                s = scores[si]
                p = preds[si]

                if probability:
                    for i, score in (enumerate(s)):
                        tag = labels.id2unit(i)
                        score = round(score.item(), 4)
                        if score > 0.001 and tag != '_SP':
                            tokprob = tokprob + '\t' + labels.id2unit(i) + ' ' + str(score)
                else:
                    tokprob = '\t' + preds[si][0]
                si = si + 1
                if lemmatize:
                    lemma = '\t' + token.words[0].lemma
            print(token.text + tokprob + lemma)
        #print('\n')
    sys.stdout = original

#tag_classla(r"C:\Users\mihailo\classla_resources\sr", r"C:\Users\mihailo\desktop\New text document.txt",r"C:\Users\mihailo\desktop\new.txt",probability, lemmatize, tokenize)