from scripts.TagAll import tag_complex
from scripts.matrixworks import matrixworks
from scripts.pipeline import lexentries, training_prep
from scripts.lexmagic import lexmagic
from scripts.conversion import convert as conv


def complex_test(tagger="", file="", lexiconmagic=True, transliterate=True, out_path="./data/output",
                 lex_path="./data/lexicon/default"):

    tempfiles = ([])
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
        words, orliglineslm = lexmagic(set(entries_u), set(entries_c), set(entries_l), words)

    for tagset in tagsets:
        modelname = tagset + ".pt"

        with open(out_path + '/temp_test', 'w', encoding='utf-8') as temp:
            temp.write('\n'.join(words))
            tempfiles.append(out_path + '/temp_test')

        tag_accus = {}

        # for a in accus:
        # tag_accus[a.split('\t')[0]] = float(a.split('\t')[1].rstrip('\n'))

        newtags, tagger_tags, probs, csv = tag_complex(tagger, "", [out_path + '/temp_test'], out_path, False, False,
                                               False, False, False, False, True, [modelname], {}, False)

        matrixworks(csv, tag_accus, newtags[modelname], tags[tagset], tagger_tags, probs[modelname], words)
