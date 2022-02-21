import sys
import pathlib
import stanza
import re
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

    tokens = stanza.Pipeline(par, dir=pardir, processors='tokenize', logging_level="FATAL")

    if tokenize:
        document = tokens.process(doc=text)
    else:
        tokens.processors["tokenize"].config["pretokenized"] = True
        sents = text.split('\n\n')
        for i, s in enumerate(sents):
            sents[i] = s.split('\n')
        document = tokens.process(doc=sents)

    lem = ""
    if lemmatize:
        lem = ",lemma"

    nlp = stanza.Pipeline(par, dir=pardir, processors='tokenize,pos'+lem, logging_level="FATAL")

    scores, preds, newdoc = getScores(nlp, document, probability, lemmatize)

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