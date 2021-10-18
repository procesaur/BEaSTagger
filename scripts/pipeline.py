import os
import re
from pathlib import Path
from scripts.tokenizer import rel_tokenize

from spacy.tokens._serialize import DocBin
from spacy.training.converters.conllu_to_docs import conllu_to_docs
from spacy.cli.convert import _write_docs_to_file


def prepare_spacy(conlulines, tempdirs, traindir, devdir):

    conllufile = "\n".join(conlulines)
    dox = [x for x in conllu_to_docs(conllufile, n_sents=10)]
    chunksize = len(dox) * 0.9
    spacy_train = ([])
    spacy_dev = ([])
    for i, doc in enumerate(dox):
        if i < chunksize:
            spacy_train.append(doc)
        else:
            spacy_dev.append(doc)

    concatdocsToFile(spacy_train, Path(traindir))
    concatdocsToFile(spacy_dev, Path(devdir))
    tempdirs.append(traindir)
    tempdirs.append(devdir)


def prepare_stanza(conlulines, tempdirs, traindir, devdir):
    conlulines = [string for string in conlulines if string != "\n"]
    chunksize = len(conlulines) * 0.9
    train_stanza = ([])
    dev_stanza = ([])
    for i, doc in enumerate(conlulines):
        if i < chunksize:
            train_stanza.append(doc)
        else:
            dev_stanza.append(doc)

    with open(traindir, 'w', encoding='utf8') as f:
        f.write('\n'.join(train_stanza))

    with open(devdir, 'w', encoding='utf8') as f:
        f.write('\n'.join(dev_stanza))

    tempdirs.append(traindir)
    tempdirs.append(devdir)


def makeconllu(lst, tagmap):
    # sentences sepparated by a double newline
    # extra carefull with quotes, as we will transform it into JSON
    # word ord number, word, lemma, ud tag, tag, and several empty values that arent necessary
    idx = 1
    for i, line in enumerate(lst):

        if line == "":
            idx = 1
            lst[i] = lst[i] + '\n'
        else:
            la = line.split('\t')
            word = la[0]
            # word = word.replace('"', '||||').replace("'", "|||")
            pos = la[1]
            if len(la) > 2:
                lemma = la[2]
            else:
                lemma = ""
            # lemma = lemma.replace('"', '||||').replace("'", "|||")
            lst[i] = str(idx) + '\t' + word + '\t' + lemma + '\t' + tagmap[pos] + '\t' + pos + '\t_\t_\t_\t_\t_'
            idx += 1
    return lst


def concatdocsToFile (docs, outpath):
    db = DocBin(docs=docs, store_user_data=True)
    data = db.to_bytes()
    _write_docs_to_file(data, outpath, "")


def chunkses(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def filechunkses(paragraphs, n, total):
    avg = round(total/n)
    le = 0
    plist = ([])
    for i, p in enumerate(paragraphs):
        plist.append(p)
        le += len(p)
        if le > avg or i+1 == len(paragraphs):
            yield plist
            le = 0
            plist = ([])


def segmentize(file="", erase_newlines=True):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            fulltext = f.read()

    except:
        with open(file, 'r', encoding='latin2') as f:
            fulltext = f.read()

    if erase_newlines:
        control = 12
    else:
        control = 11

    mpa = dict.fromkeys(range(0, control), " ")
    mpa.update(dict.fromkeys(range(12, 32), " "))
    fulltext = fulltext.translate(mpa)
    fulltext = fulltext.replace('<', '\n<').replace('>', '>\n')
    fulltext = re.sub(r' +', ' ', fulltext)
    fulltext = re.sub(r'\n+', '\n', fulltext)
    fulltext = fulltext.replace('\n \n', '\n').replace(' \n ', '\n')
    fulltext = re.sub(r'\n+', '\n', fulltext)

    return fulltext.split('\n'), len(fulltext)


def tokenize(paragraphs, out_path=""):
    lines = rel_tokenize(paragraphs, out_path, 'sr')
    os.remove(out_path + '/tempw')
    return lines
