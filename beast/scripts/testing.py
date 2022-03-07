from sklearn.metrics import classification_report, precision_recall_fscore_support as score, accuracy_score as accur
from beast.scripts.tagging import tag_complex
from beast.scripts.pipeline import lexentries, training_prep, lexmagic
from beast.scripts.conversion import convert as conv

import os
import pandas as pd


def complex_test(tagger="", file="", lexiconmagic=True, transliterate=True, full=False, confidence=0.93,
                 out_path="",  lex_path="", tt_path=""):

    words = ([])
    tags = {}

    lines, lemacol, tagsets, newline, colsn = training_prep(file)

    tagset_names = ([])

    for tagset in tagsets:
        tags[tagset] = ([])
        tagset_names.append(tagset)

    for line in lines:
        if line == '\n':
            words.append(line)
        else:
            cols = line.split("\t")
            words.append(cols[0])
            for i in range(1, colsn-1):
                tags[tagset_names[i-1]].append(cols[i])

    del lines

    # transliteration
    if transliterate:
        print('transliteration...')
        for i, line in enumerate(words):
            words[i] = conv(line, 'CYRtoLAT')

    # This procedure changes the training dataset based of a lexicon
    if lexiconmagic:
        # load all possible words from the lexicon (in uppercase, capitalized and lowercase)
        entries_u, entries_c, entries_l = lexentries(lex_path)
        # convert using lexicon magic
        words, orliglineslm, rc = lexmagic(set(entries_u), set(entries_c), set(entries_l), words)

    with open(out_path + "/temp_test", 'w', encoding='utf-8') as temp:
        temp.write('\n'.join(words))

    for tagset in tagsets:
        modelname = tagset + ".pt"
        if os.path.isfile(tagger+"/"+modelname):

            newtags, tagger_tags, probs, matrix, flat = tag_complex(tagger, "", [out_path + "/temp_test"], out_path,
                                                                    tt_path, False, False, False, False, False, False, True,
                                                                    False, [modelname], {}, False, False, False, confidence)

            test_results(tags[tagset], newtags[modelname], tagger_tags, tagset, matrix, flat, full)


def test_results(correct_tags, beast_tags, tagger_answers, tagset, matrix, flat_tagset, full=False, dump=""):
    if dump != "":
        with open(dump+"_"+tagset, 'w', encoding='utf-8') as dp:
            dp.write("corr\tbeast\t" + "\t".join(tagger_answers.keys()) +"\n")
            for i, x in enumerate(correct_tags):
                dp.write(x+"\t"+beast_tags[i])
                for k in tagger_answers.keys():
                    dp.write("\t"+tagger_answers[k][i])
                dp.write("\n")

    taggers = list(tagger_answers.keys())

    for tagger in taggers:
        if "_" + tagset.split(".")[0].lower() not in tagger.lower():
            del tagger_answers[tagger]

    delete_cols = ([])

    for i in range(0, len(flat_tagset)):
        if "_" + tagset.split(".")[0].lower() not in flat_tagset[i].lower():
            delete_cols.append(i)

    df = pd.DataFrame(data=matrix, columns=flat_tagset)
    df.drop(df.columns[delete_cols], axis=1, inplace=True)

    tagger_answers["high"] = ([])
    high = df.idxmax(axis=1)

    for x in high:
        tagger_answers["high"].append(x.split("__")[1])

    tags = ([])
    for x in df.columns:
        tags.append(x.split("__")[1])

    tags = list(set(tags))

    for tag in tags:
        cols = [col for col in df.columns if col.endswith("__" + tag)]
        df[tag] = df[cols].sum(axis=1)

    tagger_answers["jury"] = df.idxmax(axis=1).tolist()
    tagger_answers["BEaST"] = beast_tags

    if full:
        for a in tagger_answers:
            print(a)
            print(classification_report(correct_tags, tagger_answers[a], zero_division=1, digits=3))
    else:
        print("\t".join(["tagger", "prec", "rec", "f1", "acc"]))
        for a in tagger_answers:
            acc = accur(correct_tags, tagger_answers[a])
            w_prec, w_rcl, w_f, support = score(correct_tags, tagger_answers[a], average='weighted', zero_division=1)
            vals = [w_prec, w_rcl, w_f, acc]
            for i, val in enumerate(vals):
                vals[i] = str(round(val, 3))
            print(a+"\t" + "\t".join(vals))
