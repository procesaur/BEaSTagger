import spacy
import sys
from spacy.tokens import Doc
import sys
from unicodedata import category

from spacy.pipeline.tagger import Tagger


probability = True
lemmat = False
tokenize = False

chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))
numbers = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","}


def tag_spacytagger(par_path, file_path, out_path, probability, lemmat, tokenize, right = False):
    nlp = spacy.load(par_path)
    nlp.max_length = 2000000
    nom = 1
    tags = nlp.meta["labels"]["tagger"]
    sent = "SENT" in tags
    punct = "PUNCT" in tags
    num = "NUM" in tags

    if right:
        nom = -1

    pr = ""
    if probability:
        pr = " 1.0"

    def custom_tokenizer(text):
        tokens = text.split('\n')
        # tokens = list(line for line in tokens if line not in ['\n', '', '\0'])
        for i, token in enumerate(tokens):
            if token in ['\n', '', '\0']:
                tokens[i] = "{{S}}"

        return Doc(nlp.vocab, tokens)

    if not tokenize:
        nlp.tokenizer = custom_tokenizer

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        doc = nlp(content)

        tagger = nlp.get_pipe('tagger')
        scores = tagger.model.predict([doc])

        original = sys.stdout
        sys.stdout = open(out_path, 'w', encoding='utf-8')

        for idx, token in (enumerate(doc)):

            if token.text != " " and token.text != "{{S}}":

                if set(token.text).issubset(punctuation) and punct:
                    if idx + 1 == len(doc):
                        tokprob = "\tPUNCT" + pr
                    else:
                        if doc[idx+nom].text == "{{S}}" and sent:
                            tokprob = "\tSENT" + pr
                        else:
                            tokprob = "\tPUNCT" + pr

                elif set(token.text).issubset(numbers) and num:
                    tokprob = "\tNUM" + pr

                else:
                    tokprob = '\t' + token.tag_
                    if probability:
                        tokprob = ''
                        thisscores = scores[0][idx]
                        for i, score in (enumerate(thisscores)):
                            score = round(score, 4)
                            if score > 0.001 and tagger.labels[i] != '_SP':
                                tokprob = tokprob + '\t' + tagger.labels[i] + ' ' + str(score)

                lemma = ''
                if lemmat:
                    lemma = '\t' + token.lemma_

                print(token.text + tokprob + lemma)

        sys.stdout = original
