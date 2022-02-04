import os
import numpy as np
import re
import pandas as pd

from beast.scripts.conversion import convert as conv
from beast.TreeTagger.treetagger import tag_treetagger
from beast.SpacyTagger.spacyworks import tag_spacytagger
# from Classla.TagClassla import tag_classla
from beast.scripts.torchworks import test_prob_net
from beast.scripts.pipeline import segmentize, big_chunkus, rem_xml, write_chunks, lexmagic
from beast.scripts.tokenizer import rel_tokenize
from beast.scripts.pipeline import probtagToMatrix, get_taggers, lemmas_dic, lexentries


def tag_complex(par_path, lex_path, file_paths, out_path, tt_path, lexiconmagic=False, transliterate=True,
                tokenization=True, MWU=False, onlyPOS=False, lemmat=False, testing=False, quiet=True,
                models=[], lemmatizers={}, lempos=False, probability=False, stdout=False, confidence=0.93):

    # default parameters
    tempfiles = ([])

    # get a name for our model
    par_name = os.path.basename(os.path.normpath(par_path))
    if ' ' in par_name:
        par_name = par_name.split(" ")[-1]

    # lexicon magic doesn't work without lexicon
    if lex_path == "":
        lexiconmagic = False

    # mwu doesn't work without tokenization
    if not tokenization:
        MWU = False

    if not quiet:
        print("preparation")
    if lexiconmagic:
        # load all possible words from the lexicon (in uppercase, capitalized and lowercase)
        entries_u, entries_c, entries_l = lexentries(lex_path)

    if lemmat:
        # load lemmas for each word and POS from lexicon
        lemdic = lemmas_dic(lemmatizers)

    for filex in file_paths:
        # split files into smaller if they are over the terminal size.
        # this returns a list of new files to tag and their map to the original files
        files, filesmap = big_chunkus(filex, out_path, quiet)

    # pipeline for each file VVV
    for file in files:

        # initialize file
        newtags = {}
        newprobs = {}

        for model in models:
            newprobs[model] = ([])
            newtags[model] = ([])

        paragraphs, total, size = segmentize(file)

        # tokenization
        if tokenization:
            if not quiet:
                print('tokenization...')
            origlines = rel_tokenize(paragraphs, out_path)
        else:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    origlines = f.readlines()
            except:
                with open(file, 'r', encoding='latin2') as f:
                    origlines = f.readlines()

        # exclude xml from file, but wirte it down to add later
        lines, origlines, exclusion = rem_xml(origlines)

        # transliteration
        if transliterate:
            if not quiet:
                print('transliteration...')
            for i, line in enumerate(lines):
                lines[i] = conv(line, 'CYRtoLAT')

        # This procedure changes the training dataset based of a lexicon
        if lexiconmagic:
            # convert using lexicon magic
            lines, orliglineslm, rc = lexmagic(set(entries_u), set(entries_c), set(entries_l), lines)
            if not quiet:
                print("lexmagic replacements: " + str(rc))

        # write lines into file for tagging - chunking, and get a total list of targets
        targets = write_chunks(lines, out_path)

        # write additional tempfiles
        for t in targets:
            tempfiles.append(t)
        tempfiles.append(out_path + '/tempx2')

        # getting a list of taggers
        taggers_array = get_taggers(par_path)

        if not quiet:
            print("tagging with " + str(len(taggers_array)) + " taggers...")

        for tr in targets:

            matrices = ([])
            tagaccus = ([])
            tagsets = ([])
            lemmas = {}

            tagger_tags = {}

            for tagger in taggers_array:
                tlines = tag_any(tr, par_path + '/' + tagger, out_path, tt_path)
                mat, accu, tagset, tags = probtagToMatrix(tlines, tagger.split('/')[-1])
                matrices.append(mat)
                tagaccus.append(accu)
                tagsets.append(tagset)

                tagger_tags[tagger.split('/')[-1]] = tags

            flat_tagset = [item for sublist in tagsets for item in sublist]

            matricout = np.concatenate(matrices, axis=0)
            csv = out_path + "/matrix-prob_tag.csv"
            with open(csv, 'w', encoding='utf-8') as m:
                m.write('\t'.join(flat_tagset) + '\n')
                for idx, line in enumerate(matricout.transpose()):
                    # m.write(words[idx] + '\t') we can write the words but dont need to
                    # m.write(tags[idx] + '\t')
                    np.savetxt(m, line[np.newaxis], delimiter='\t', fmt='%.4f')

            for model in models:
                newtagsx, newprobsx = test_prob_net(csv, par_path, out_path, model)

                tagger_answers = tagger_tags.copy()
                taggers = list(tagger_answers.keys())

                for tagger in taggers:
                    if "_" + model.split(".")[0].lower() not in tagger.lower():
                        del tagger_answers[tagger]

                if tagger_answers:
                    delete_cols = ([])

                    for i in range(0, len(flat_tagset)):
                        if "_" + model.split(".")[0].lower() not in flat_tagset[i].lower():
                            delete_cols.append(i)

                    df = pd.DataFrame(data=matricout.transpose(), columns=flat_tagset)
                    df.drop(df.columns[delete_cols], axis=1, inplace=True)

                    tagger_answers["high"] = ([])
                    high = df.idxmax(axis=1)

                    for x in high:
                        tagger_answers["high"].append(x.split("__")[1])

                    for i, tag in enumerate(newtagsx):
                        if newprobsx[i] < confidence:
                            newtagsx[i] = tagger_answers["high"][i]

                newtags[model].extend(newtagsx)
                newprobs[model].extend(newprobsx)

        if not testing:
            tempfiles.append(csv)

        taggedlines = ([])


        # if there is lemmatization
        if lemmat:
            if not quiet:
                print('lemmatizing')

            noslines = list(line.rstrip('\n') for line in lines if line not in ['\n', ''])
            for model in models:
                mo_del = model.split(".")[0]
                lmodel = ""
                if mo_del in lemmatizers.keys():
                    lmodel = mo_del
                if mo_del.lower() in lemmatizers.keys():
                    lmodel = mo_del.lower()
                if mo_del.upper() in lemmatizers.keys():
                    lmodel = mo_del.upper()
                
                if lmodel != "":
                    lemmas[model] = ([])
                    if ".par" in lemmatizers[lmodel]:
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
                                lemmas[model].append(lemdic[lmodel][word][newtags[model][i]].rstrip())
                            except:
                                lemmas[model].append(word)
            # print(lemmas.keys())
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
                    if probability:
                        taggedline += "\t" + str(round(newprobs[model][i], 3))
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

        header = "word\t"
        for x in models:
            xx = "\t" + x
            if x in lemmas.keys:
                xx += "\tlemma_" + x
                if lempos:
                    xx += "\tlempos_" + x
            if probability:
                xx += "\tprobability"
            header += xx

        if not testing:
            if stdout:
                print(header)
                for line in finalines:
                    print(line.rstrip('\n'))
            else:
                if os.path.isfile(filesmap[file]):
                    writepath = out_path + '/' + os.path.basename(filesmap[file]) + "_" + par_name + ".tt"
                else:
                    writepath = out_path + "/input_" + par_name + ".tt"
                with open(writepath, 'a+', encoding='utf-8') as m:
                    m.write(header)
                    for line in finalines:
                        m.write(line)

        del taggedlines

        if filesmap[file] != file:
            os.remove(file)
        # remove temp files
        for tempf in tempfiles:
            if os.path.isfile(tempf):
                os.remove(tempf)

    return newtags, tagger_tags, newprobs, matricout.transpose(), flat_tagset


def tag_any(file, par_path, out_path, tt_path):

    tempfiles = ([])

    filename, file_extension = os.path.splitext(par_path)

    isright = False
    if 'right' in par_path:
        isright = True

    # file read
    with open(file, 'r', encoding='utf-8') as f:
        origlines = f.readlines()

    lines = [line.rstrip('\n') for line in origlines if not re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line)]

    # reverse lines if its a right tagger
    if isright:
        lines.reverse()

    for i, line in enumerate(lines):
        lines[i] = line.rstrip('\n')

    # write lines into file for tagging
    targets = ([])
    with open(out_path + '/temp2', 'w', encoding='utf-8') as temp:
        temp.write('\n'.join(lines))
    tempfiles.append(out_path + '/temp2')
    targets.append(out_path + '/temp2')

    newlines = ([])

    # use tagging procedures
    for fx in targets:
        # if treetagger
        if file_extension == '.par':
            tag_treetagger(par_path, fx, out_path + '/temp3', True, False, tt_path)

        # if classla
        # elif par_path.endswith("/sr"):
          #   tag_classla(par_path, fx, out_path + '/temp3', probability, lemmat, False)

        # if spacy tagger
        else:
            if isright:
                tag_spacytagger(par_path, fx, out_path + '/temp3', True, False, False, True)
            else:
                tag_spacytagger(par_path, fx, out_path + '/temp3', True, False, False)

        with open(out_path + '/temp3', 'r', encoding='utf-8') as f:
            newlinesx = f.readlines()
            newlines += list(line for line in newlinesx if line != '\n')

    tempfiles.append(out_path + '/temp3')

    if isright:
        newlines.reverse()

    # remove temp files
    for tempf in tempfiles:
        if os.path.isfile(tempf):
            os.remove(tempf)

    del lines
    del origlines

    return newlines
