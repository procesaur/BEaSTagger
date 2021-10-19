import os
import re
import math
import random
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


def lexentries(lex_path):
    with open(lex_path, 'r', encoding='utf-8') as lex:
        entriesfull = [wordx.split('\t')[0] for wordx in lex.readlines()]

    entries_c = [wordx for wordx in entriesfull if wordx[0].isupper()]
    entries_l = [wordx for wordx in entriesfull if not wordx[0].isupper()]
    entries_u = [wordx for wordx in entries_c if wordx.isupper()]
    entries_c += [wordx for wordx in entries_c if not wordx.isupper()]

    del entriesfull
    return entries_u, entries_c, entries_l


def lemmas_dic(lemmatizers):
    lemdic = {}
    for modell in lemmatizers.keys():
        lemdic[modell] = {}
        with open(lemmatizers[modell], 'r', encoding='utf-8') as d:
            diclines = d.readlines()

        for d in diclines:
            tabs = d.split("\t")
            ent = tabs[0]
            lemdic[modell][ent] = {}
            del tabs[0]
            for t in tabs:
                try:
                    lemdic[modell][ent][t.split(" ")[0]] = t.split(" ")[1].rstrip()
                except:
                    pass
    return lemdic


def big_chunkus(filex, out_path, terminal_size=50000000):
    files = ([])
    filesmap = {}

    paragraphs, total = segmentize(filex)
    fn = math.ceil(os.path.getsize(filex) / terminal_size)
    if len(paragraphs) < fn + 1:
        fn = len(paragraphs)
    if fn > 1:
        print("huge file - splitting")
        filechunks = filechunkses(paragraphs, fn, total)
        print("writing " + str(fn) + " chunks to disk")

        for i, c in enumerate(filechunks):
            if i != fn:
                fname = out_path + "/" + os.path.basename(filex) + '___' + str(i)
                with open(fname, 'w', encoding='utf-8') as temp:
                    temp.write('\n'.join(c))
                files.append(fname)
                filesmap[fname] = filex
            else:
                fname = out_path + "/" + os.path.basename(filex) + '___' + str(i - 1)
                with open(fname, 'a', encoding='utf-8') as temp:
                    temp.write('\n'.join(c))

        del filechunks

    else:
        files.append(filex)
        filesmap[filex] = filex

    del paragraphs
    return files, filesmap


def rem_xml(lines):
    exclusion = {}
    noslines = list(line.rstrip('\n') for line in lines if line not in ['\n', ''])

    for idx, line in enumerate(noslines):
        if re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line):
            exclusion[idx] = line
    del noslines

    origlines = [line.rstrip('\n') for line in lines if not re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line)]

    newlines = origlines.copy()
    origlines = list(line.rstrip('\n') for line in origlines if line not in ['\n', '', '\0'])

    return newlines, origlines, exclusion


def write_chunks(lines, out_path, chunklines=80000, testing=False, results=None):
    targets = ([])
    if len(lines) < chunklines or testing:
        with open(out_path + '/tempx2', 'w', encoding='utf-8') as temp:
            temp.write('\n'.join(lines))
        targets.append(out_path + '/tempx2')
    else:
        print('chunking...')

        if results is None:
            alltext = '\n'.join(lines)
            alltext = alltext.rstrip('\n')
            sents = alltext.split('\n\n')
            for i, s in enumerate(sents):
                sents[i] = s + '\n'
        else:
            for i, w in enumerate(lines):
                if results[i] == 'SENT':
                    lines[i] = w + 'SENT'

            alltext = '\n'.join(lines)
            alltext = alltext.rstrip('SENT')
            sents = alltext.split('SENT')

        chunkn = round(len(lines) / chunklines)
        chunkovi = chunkses(sents, round(len(sents) / chunkn))

        del alltext
        del sents

        for i, c in enumerate(chunkovi):
            with open(out_path + '/prepared' + str(i), 'w', encoding='utf-8') as temp:
                temp.write('\n'.join(c))
            targets.append(out_path + '/prepared' + str(i))

        del chunkovi
        print(str(len(targets)) + " chunks created")
    return targets


def get_taggers(path):
    taggers_array = ([])
    taggers_arr = os.listdir(path)

    par_file = ""
    for t in taggers_arr:
        if '.par' in t or 'Spacy' in t:
            taggers_array.append(t)
        if '.par' in t and par_file == "":
            par_file = t
        if 'TreeTagger.par' in t:
            par_file = t
        if t.endswith('sr'):
            taggers_array.append(t)

    return taggers_array


def ratio_split(ratio, lines):
    count_sen = 0
    # remove new lines form the end of each line, because they are an array now
    for i, line in enumerate(lines):
        lines[i] = line.rstrip('\n')
        # check if line is empty, increase, sentence counter
        if lines[i] == "":
            count_sen += 1

    # check if text is split to sentences (is there count_sen), if not try to split by SENT tag.
    if count_sen < 1:
        # now add newline after each SENT, we are trying to break text into sentences
        for i, line in enumerate(lines):
            if 'SENT' in line:
                lines[i] = lines[i] + '\n'
    # create sentence array
    text = re.sub(r'\n\n+', '\n\n', '\n'.join(lines)).strip()
    sentences = text.split('\n\n')
    # we set chunk sizes to X% of text - this will be used for training
    chunksize = len(sentences) * ratio

    # strip any extra newlines
    for i, sent in enumerate(sentences):
        if sent.endswith('\n'):
            sentences[i] = sent.rstrip('\n')

    # randomly shuffle sentences to get unbiased corpora
    random.shuffle(sentences)
    # initialize are chunks
    train = ([])
    tune = ([])
    # add first 90% of sentences to "ninety" array, and the rest to "ten" array
    for i, sent in enumerate(sentences):
        if i < chunksize:
            train.append(sent + "\n\n")
        else:
            tune.append(sent + "\n\n")

    return train, tune


def training_prep(file):
    tagsets = {}
    with open(file, 'r', encoding='utf-8') as fl:
        lines = fl.readlines()  # all lines including the blank ones

    meta = lines[0].rstrip().split("\t")
    colsn = len(meta)
    del lines[0]

    newline = ""
    for i in range(1, colsn):
        newline += "\t"
    newline += "\n"

    lemacol = -1
    for i, m in enumerate(meta):
        if i > 0:
            if m != "lemma" and m != "lema":
                tagsets[m] = i
            else:
                lemacol = i

    for i, line in enumerate(lines):
        if line in ['\n', '', '\0', newline]:
            lines[i] = '\n'

    return lines, lemacol, tagsets, newline, colsn
