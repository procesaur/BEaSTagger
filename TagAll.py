import os
import ntpath
import math
import re

from tkinter import filedialog
from tkinter import *
from tqdm import tqdm
import numpy as np
from ftfy import fix_text

from scripts.conversion import convert as conv
from scripts.lexmagic import lexmagic
from TreeTagger.tag import tag_treetagger
from SpacyTagger.TagSpacy import tag_spacytagger
from Classla.TagClassla import tag_classla
from scripts.matrixworks import probtagToMatrix, test_prob_net
from scripts.pipeline import tokenize, chunkses, segmentize, filechunkses


def tag_complex(par_path="", lex_path="", file_paths="", out_path="", lexiconmagic=False, transliterate=True,
                tokenization=True, MWU=False, onlyPOS=False, results=None, lemmat=False, testing=False,
                models=["net-prob.pt"], lemmatizers={}, lempos=False):
    tempfiles = ([])
    entries_u = ([])
    entries_c = ([])
    entries_l = ([])
    entriesfull = ([])
    lemdic = {}

    Tk().withdraw()
    chunklines = 80000
    term_size = 50000000
    # setting variables if unset
    if file_paths == "":
        file_paths = filedialog.askopenfilenames(initialdir="./data/training",
                                                 title="Select text files",
                                                 filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert"),
                                                            ("all files", "*.*")))
    if par_path == "":
        par_path = filedialog.askdirectory(initialdir="./data/output", title="Select complex model directory")

    if out_path == "":
        out_path = filedialog.askdirectory(initialdir="./data/output", title="Select output directory")

    if lex_path == "" and lexiconmagic:
        lex_path = filedialog.askopenfilename(initialdir="./data/lexicon",
                                              title="Select lexicon file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    if lex_path == "":
        lexiconmagic = False

    print("preparation")
    if lexiconmagic:
        # laod all possible words from the lexicon
        with open(lex_path, 'r', encoding='utf-8') as lex:
            entriesfull += [wordx.split('\t')[0] for wordx in lex.readlines()]
        entries_c += [wordx for wordx in entriesfull if wordx[0].isupper()]
        entries_l += [wordx for wordx in entriesfull if not wordx[0].isupper()]
        entries_u += [wordx for wordx in entries_c if wordx.isupper()]
        entries_c += [wordx for wordx in entries_c if not wordx.isupper()]

        del entriesfull
    for modell in lemmatizers.keys():
        lemdic[modell] = {}
        if lemmat and ".par" not in lemmatizers[modell]:
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

    if not tokenization:
        MWU = False

    par_name = os.path.basename(os.path.normpath(par_path))
    if ' ' in par_name:
        par_name = par_name.split(" ")[-1]

    files = ([])
    filesmap = {}
    for filex in file_paths:

        paragraphs, total = segmentize(filex)
        fn = math.ceil(os.path.getsize(filex)/term_size)
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
                    fname = out_path + "/" + os.path.basename(filex) + '___' + str(i-1)
                    with open(fname, 'a', encoding='utf-8') as temp:
                        temp.write('\n'.join(c))

            del filechunks

        else:
            files.append(filex)
            filesmap[filex] = filex

        del paragraphs

    for file in files:
        newtags = {}
        newprobs = {}
        for model in models:
            newprobs[model] = ([])
            newtags[model] = ([])

        paragraphs, total = segmentize(file)

        # tokenization
        if tokenization:
            print('tokenization...')

            origlines = tokenize(paragraphs, MWU, out_path, par_path)
        else:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    origlines = f.readlines()

            except:
                with open(file, 'r', encoding='latin2') as f:
                    fulltext = fix_text(f.read())
                    origlines = fulltext.split('\n')

        exclusion = {}
        noslines = list(line.rstrip('\n') for line in origlines if line not in ['\n', ''])

        for idx, line in enumerate(noslines):
            if re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line):
                exclusion[idx] = line
        del noslines

        origlines = [line.rstrip('\n') for line in origlines if not re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line)]

        lines = origlines.copy()
        origlines = list(line.rstrip('\n') for line in origlines if line not in ['\n', '', '\0'])

        # transliteration
        if transliterate:
            print('transliteration...')
            for i, line in enumerate(lines):
                lines[i] = conv(line, 'CYRtoLAT')

        # Some lexicon magic (might take a while, based on file sizes)
        # This procedure changes the training dataset based of a lexicon
        if lexiconmagic:
            # if there isn't already lm file loaded
            # convert using lexicon magic
            lines, orliglineslm = lexmagic(set(entries_u), set(entries_c), set(entries_l), lines)

        # write lines into file for tagging - chunking
        targets = ([])
        if len(lines) < chunklines or testing:
            with open(out_path + '/tempx2', 'w', encoding='utf-8') as temp:
                temp.write('\n'.join(lines))
            tempfiles.append(out_path + '/tempx2')
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
            chunkovi = chunkses(sents, round(len(sents)/chunkn))

            del alltext
            del sents

            for i, c in enumerate(chunkovi):
                with open(out_path + '/prepared' + str(i), 'w', encoding='utf-8') as temp:
                    temp.write('\n'.join(c))
                tempfiles.append(out_path + '/prepared' + str(i))
                targets.append(out_path + '/prepared' + str(i))

            del chunkovi
            print(str(len(targets)) + " chunks created")
        # getting a list of taggers
        taggers_arr = os.listdir(par_path)
        taggers_array = ([])
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

        taggedlinesall = ([])

        print("tagging with " + str(len(taggers_array)) + " taggers...")
        for i, tr in tqdm(enumerate(targets)):

            tlines = ([])
            matrices = ([])
            tagaccus = ([])
            tagsets = ([])
            lemmas = {}

            tagger_tags = {}

            for tagger in taggers_array:
                # print(tagger)
                tlines = tag_any(False, False, False, False, True, False, False, [tr], par_path + '/' + tagger,
                                  out_path, False, "", False)

                mat, accu, tagset, tags = probtagToMatrix(tlines, tagger.split('/')[-1])
                matrices.append(mat)
                tagaccus.append(accu)
                tagsets.append(tagset)

                tagger_tags[tagger.split('/')[-1]] = tags

            flat_tagset = [item for sublist in tagsets for item in sublist]

            tlines = ([])

            if False:
                for i in range(0,len(matrices)):
                    with open(out_path + "/matrix-prob_tag"+str(i)+".csv", 'w', encoding='utf-8') as m:
                        m.write('\t'.join(flat_tagset) + '\n')
                        for idx, line in enumerate(matrices[i].transpose()):
                            # m.write(words[idx] + '\t') we can write the words but dont need to
                            # m.write(tags[idx] + '\t')
                            np.savetxt(m, line[np.newaxis], delimiter='\t', fmt='%.4f')
                    tempfiles.append(out_path + "/matrix-prob_tag"+str(i)+".csv")

            # print("building result martix...")
            matricout = np.concatenate(matrices, axis=0)
            with open(out_path + "/matrix-prob_tag.csv", 'w', encoding='utf-8') as m:
                m.write('\t'.join(flat_tagset) + '\n')
                for idx, line in enumerate(matricout.transpose()):
                    # m.write(words[idx] + '\t') we can write the words but dont need to
                    # m.write(tags[idx] + '\t')
                    np.savetxt(m, line[np.newaxis], delimiter='\t', fmt='%.4f')

            for model in models:
                newtagsx, newprobsx = test_prob_net(out_path + "/matrix-prob_tag.csv", par_path, out_path, model)
                newtags[model].extend(newtagsx)
                newprobs[model].extend(newprobsx)

        if not testing:
            tempfiles.append(out_path + "/matrix-prob_tag.csv")

        # if there is lemmatization and if it is possible (par file found)
        if lemmat:
            taggedlines = ([])
            print('lemmatizing')

            noslines = list(line.rstrip('\n') for line in lines if line not in ['\n', ''])

            for model in models:
                if model in lemmatizers.keys():
                    lemmas[model] = ([])
                    if ".par" in lemmatizers[model]:
                        with open(out_path + "/temp.tag", 'w', encoding='utf-8') as m:
                            for i, word in enumerate(noslines):
                                m.write(word + '\t' + newtags[model][i] + '\n')

                        tag_treetagger(lemmatizers[model], out_path + "/temp.tag", out_path + "/temp2.tag", False, True)

                        with open(out_path + "/temp2.tag", 'r', encoding='utf-8') as f:
                            taggedlines = f.readlines()
                        lemmas[model] = list(lt.rstrip('\n').split("\t")[2] for lt in taggedlines if lt not in ['\n', ''])

                    else:
                        for i, word in enumerate(noslines):
                            try:
                                lemmas[model].append(lemdic[model][word][newtags[model][i]].rstrip())
                            except:
                                lemmas[model].append(word)

            for i, l in enumerate(origlines):
                taggedline = l
                for model in models:
                    if onlyPOS:
                        taggedline += "\t" + newtags[model][i].split(':')[0]
                    else:
                        taggedline += "\t" + newtags[model][i]
                    if model in lemmas.keys():

                        taggedline += "\t" + lemmas[model][i]
                        if lempos:
                            taggedline += "\t" + lemmas[model][i] + "_" + newtags[model][i]
                taggedline += "\n"
                taggedlines.append(taggedline)

            tempfiles.append(out_path + "/temp.tag")
            tempfiles.append(out_path + "/temp2.tag")

        else:

            for i, word in enumerate(origlines):
                taggedlines.append(word)
                for model in models:
                    taggedlines.append("\t" + newtags[model][i])
                taggedlines.append("\n")

        if tokenization:
            finalines = ([])
            c = 0
            for i in range(0, len(taggedlines)+len(exclusion)):
                if i in exclusion.keys():
                    finalines .append(exclusion[i]+"\n")
                else:
                    finalines.append(taggedlines[c])
                    c += 1
        else:
            finalines = taggedlines

        with open(out_path + '/' + os.path.basename(filesmap[file]) + "_" + par_name + ".tt", 'a+', encoding='utf-8') as m:
            for line in finalines :
                m.write(line)

        del taggedlines
        if filesmap[file] != file:
            os.remove(file)
        # remove temp files
        for tempf in tempfiles:
            if os.path.isfile(tempf):
                os.remove(tempf)

    return newtags, tagger_tags, newprobs


def tag_any(transliterate, lexiconmagic, tokenization, MWU, probability, onlyPOS,
            lemmat, file_paths, par_path, out_path, output, lex_path="", chunking=True):
    tempfiles = ([])
    entries_u = ([])
    entries_c = ([])
    entries_l = ([])
    entriesfull = ([])

    chunklines = 100000
    if lex_path == "":
        lexiconmagic = False

    if not tokenization:
        MWU = False

    if probability:
        lemmat = False

    filename, file_extension = os.path.splitext(par_path)

    if lexiconmagic:
        # laod all possible words from the lexicon
        with open(lex_path, 'r', encoding='utf-8') as lex:
            entriesfull += [wordx.split('\t')[0] for wordx in lex.readlines()]
        entries_c += [wordx for wordx in entriesfull if wordx[0].isupper()]
        entries_l += [wordx for wordx in entriesfull if not wordx[0].isupper()]
        entries_u += [wordx for wordx in entries_c if wordx.isupper()]
        entries_c += [wordx for wordx in entries_c if not wordx.isupper()]
        del entriesfull

    for file in file_paths:

        isright = False
        if 'right' in par_path:
            isright = True

        # file read
        if tokenization:
            print('tokenization...')
            origlines = tokenize(file, MWU, out_path, par_path)
        else:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    origlines = f.readlines()

            except:
                print('warning! mixed encoding...')
                with open(file, 'r', encoding='latin2') as f:
                    fulltext = fix_text(f.read())
                    origlines = fulltext.split('\n')

        exclusion = {}

        noslines = origlines.copy()
        noslines = list(line.rstrip('\n') for line in noslines if line not in ['\n', ''])

        for idx, line in enumerate(noslines):
            if re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line):
                exclusion[idx] = line

        del noslines

        origlines = [line.rstrip('\n') for line in origlines if not re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line)]

        lines = origlines.copy()
        origlines = list(line for line in origlines if line != '\n' and line != '')

        # transliteration
        if transliterate:
            print('transliteration...')
            for i, line in enumerate(lines):
                lines[i] = conv(line, 'CYRtoLAT')

        # Some lexicon magic (might take a while, based on file sizes)
        # This procedure changes the training dataset based of a lexicon
        if lexiconmagic:
            lines, orliglineslm = lexmagic(entries_u, entries_c, entries_l, lines)
            del orliglineslm

        # reverse lines if its a right tagger
        if isright:
            lines.reverse()

        for i, line in enumerate(lines):
            lines[i] = line.rstrip('\n')

        # write lines into file for tagging

        targets = ([])
        if len(lines) < chunklines or not chunking:
            with open(out_path + '/temp2', 'w', encoding='utf-8') as temp:
                temp.write('\n'.join(lines))
            tempfiles.append(out_path + '/temp2')
            targets.append(out_path + '/temp2')
        else:
            print('chunking...')

            alltext = '\n'.join(lines)
            alltext = alltext.rstrip('\n')
            sents = alltext.split('\n\n')
            for i, s in enumerate(sents):
                sents[i] = s + '\n'

            chunkn = round(len(lines) / chunklines)
            chunkovi = chunkses(sents, round(len(sents)/chunkn))

            del alltext
            del sents

            for i, c in enumerate(chunkovi):
                with open(out_path + '/prepared' + str(i), 'w', encoding='utf-8') as temp:
                    temp.write('\n'.join(c))
                tempfiles.append(out_path + '/prepared' + str(i))
                targets.append(out_path + '/prepared' + str(i))

            del chunkovi
            print(str(len(targets)) + " chunks created")
        newlines = ([])

        # use tagging procedures
        if output:
            print("tagging...")
        for fx in targets:
            # if treetagger
            if file_extension == '.par':
                tag_treetagger(par_path, fx, out_path + '/temp3', probability, lemmat)

            # if classla
            elif par_path.endswith("/sr"):
                tag_classla(par_path, fx, out_path + '/temp3', probability, lemmat, False)

            # if spacy tagger
            else:
                if isright:
                    tag_spacytagger(par_path, fx, out_path + '/temp3', probability, lemmat, False, True)
                else:
                    tag_spacytagger(par_path, fx, out_path + '/temp3', probability, lemmat, False)

            with open(out_path + '/temp3', 'r', encoding='utf-8') as f:
                newlinesx = f.readlines()
                newlines += list(line for line in newlinesx if line != '\n')

        tempfiles.append(out_path + '/temp3')

        # format words back to original, and format pos to basic if needed

        if isright:
            newlines.reverse()

        if len(origlines) == len(newlines) and output and (transliterate or lexiconmagic or onlyPOS):
            print("finalizing > " + str(len(newlines)) + " lines...")
            finallines = ([])

            for (line, origline) in zip(newlines, origlines):
                if '\t' in line:
                    word = origline.rstrip()

                    if probability:
                        pos = line.split('\t', 1)[1].rstrip().lstrip('\t')
                        if onlyPOS:
                            posd = {}
                            poss = pos.split('\t')
                            for p in poss:
                                key = p.split(' ')[0].split(':')[0]
                                if key not in posd.keys():
                                    posd[key] = 0.00
                                posd[key] += float(p.split(' ')[1])

                            posstr = ""
                            for key in posd.keys():
                                posstr += key + " " + str(round(posd[key], 4)) + '\t'
                            posstr.rstrip('\t')

                            pos = posstr
                    else:
                        pos = line.split('\t')[1].rstrip()
                        if onlyPOS:
                            pos = pos.split(':')[0]

                    lemma = ""
                    # if treetagger also format lemma
                    if lemmat:
                        lemma = line.split('\t')[2].rstrip()

                    finallines.append(word + '\t' + pos + '\t' + lemma + '\n')

        else:
            finallines = newlines.copy()

        for i in exclusion:
            finallines.insert(i, exclusion[i] + '\n')

        if output:
            outhpath = out_path + '/' + ntpath.basename(file) + '_' + ntpath.basename(par_path) + '.tt'

            # write into final destination
            with open(outhpath, 'w', encoding='utf-8') as temp:
                temp.write(''.join(finallines))

        # remove temp files
        for tempf in tempfiles:
            if os.path.isfile(tempf):
                os.remove(tempf)

        del exclusion
        del lines
        del origlines
        del newlines

        return finallines
