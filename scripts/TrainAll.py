import os
import numpy as np

from pathlib import Path
from distutils.dir_util import remove_tree

from scripts.pipeline import prepare_spacy, lexentries, makeconllu, ratio_split
# from scripts.pipeline import prepare_stanza
from scripts.conversion import convert as conv
from scripts.lexmagic import lexmagic
from scripts.matrixworks import probtagToMatrix, train_prob_net
from scripts.TagAll import tag_any

# from Classla.TrainClassla import train_stanza
from SpacyTagger.spacyworks import train_spacy
from SpacyTagger.spacyworks import gettagmap
from TreeTagger.treetagger import train_treetagger


spacy_traindir = "/train.spacy"
spacy_devdir = "/dev.spacy"
spacyR_traindir = "/trainR.spacy"
spacyR_devdir = "/devR.spacy"
# stanza_traindir = "/train.stanza"
# stanza_devdir = "/dev.stanza"
# stanzaR_traindir = "/trainR.stanza"
# stanzaR_devdir = "/devR.stanza"

# train in both directions (left context and right context taggers)
bider = True
# transfer all cyrilic to latin
transliterate = True
# train TreeTagger
treetagger = True
# train Spacy tagger
spacytagger = True
# train Stanza
stanzatagger = False
# slit ratio (0 to 1)
ratio = 0.85
# use gpu
spacygpu = 1  # -1 for cpu
# skip to composite
notrain = False
hasfiles = False

tempfiles = ([])
tempdirs = ([])

# TreeTagger training parametres
cl = 3  # context length
dtg = 0.1  # decision tree gain
ecw = 0.2  # eq. class weight
atg = 1.1  # affix tree gain
sw = 16  # smoothing weight
parameters = ['-cl ' + str(cl),
              '-dtg ' + str(dtg),
              '-ecw ' + str(ecw),
              '-atg ' + str(atg),
              '-sw ' + str(sw),
              '-lt 0.001 -quiet']


def train_taggers(lines, out_path, lex_path, name, newdir, lexiconmagic=True):

    global tempfiles
    global tempdirs

    # transliteration
    if transliterate:
        print('transliteration...')
        for i, line in enumerate(lines):
            lines[i] = conv(line, 'CYRtoLAT')

    # This procedure changes the training dataset based of a lexicon
    if lexiconmagic:
        # load all possible words from the lexicon (in uppercase, capitalized and lowercase)
        entries_u, entries_c, entries_l = lexentries(lex_path)
        # convert using lexicon magic
        lines, orliglineslm = lexmagic(set(entries_u), set(entries_c), set(entries_l), lines)
        # save lexicon magic

    train, tune = ratio_split(ratio, lines)

    # now read the lines from the "train" set, we will use this for training from now on
    lines = ''.join(train).rstrip('\n').split('\n')

    with open(out_path + "/tune" + name, 'w', encoding='utf8') as tf:
        tf.write(''.join(tune).rstrip('\n'))

    # Create inverted set > copy "lines" and reverse their order
    if bider:
        rlines = lines.copy()
        rlines.reverse()

    print("preparing...")
    # Treetagger files preparation
    if treetagger:
        # for treetagger we save lines as is, separated by newlines
        with open(out_path + '/TreeTagger_train', 'w', encoding='utf-8') as tr:
            tr.write('\n'.join(lines))
            tempfiles.append(out_path + '/TreeTagger_train')
        # same for right context
        if bider:
            with open(out_path + '/TreeTaggerR_train', 'w', encoding='utf-8') as tr:
                tr.write('\n'.join(rlines))
                tempfiles.append(out_path + '/TreeTaggerR_train')

    # tagmap creation
    # initialize pos array
    poses = ([])
    # append pos from each line to the array
    for idx, line in enumerate(lines):
        if line != "":
            poses.append(line.split('\t')[1])
    # then just create unique set from it
    uniquepos = list(set(poses))

    # for treetagger we can save it as openclass if dont have a full one already
    if treetagger:
        with open(out_path + '/TreeTagger_openclass', 'w', encoding='utf8') as f:
            f.write('\n'.join(uniquepos))
        oc_path = out_path + '/TreeTagger_openclass'
        tempfiles.append(oc_path)

    # SpacyTagger files preparation - Stanza file preparation
    if spacytagger or stanzatagger:

        # create tagmap using array of unique pos. this is a spacy necessity
        tagmap = gettagmap(uniquepos)
        tag_map = {}
        for tag in tagmap:
            tag_map[tag] = {'pos': tagmap[tag]}
        # save tagmap in a file. this is a spacy necessity
        with open(out_path + '/SpacyTagger_tagmap', 'w', encoding='utf8') as f:
            f.write(str(tag_map).replace("\"", "\\\"").replace("'", "\""))

        tempfiles.append(out_path + '/SpacyTagger_tagmap')

        # copy the lines, as preporcessing is necessary again
        conlulines = lines.copy()
        # transfer into conllu format that is usable.
        conlulines = makeconllu(conlulines, tagmap)

        if spacytagger:
            prepare_spacy(conlulines, tempdirs, out_path + spacy_traindir, out_path + spacy_devdir)
            tempfiles.append(out_path + spacy_traindir)
            tempfiles.append(out_path + spacy_devdir)

        # if stanzatagger:
        #     prepare_stanza(conlulines, tempdirs, out_path + stanza_traindir, out_path + stanza_devdir)
        #     tempfiles.append(out_path + stanza_traindir)
        #     tempfiles.append(out_path + stanza_devdir)

        if bider:
            rconlulines = rlines.copy()
            # create conllu format
            rconlulines = makeconllu(rconlulines, tagmap)

            if spacytagger:
                prepare_spacy(rconlulines, tempdirs, out_path + spacyR_traindir, out_path + spacyR_devdir)
                tempfiles.append(out_path + spacyR_traindir)
                tempfiles.append(out_path + spacyR_devdir)

            # if stanzatagger:
            #    prepare_stanza(conlulines, tempdirs, out_path + stanzaR_traindir, out_path + stanzaR_devdir)

    # create output dirs on the disk
    if not os.path.isdir(newdir):
        os.mkdir(newdir)

    # TreeTagger training
    if treetagger:
        print("training TreeTagger")
        tt_in_path = out_path + '/TreeTagger_train'
        tt_out_path = newdir + '/TreeTagger' + name + '_right.par'
        train_treetagger(parameters, lex_path, oc_path, tt_in_path, tt_out_path)

        if bider:
            tt_in_path = out_path + '/TreeTaggerR_train'
            tt_out_path = newdir + '/TreeTagger' + name + '_right.par'
            train_treetagger(parameters, lex_path, oc_path, tt_in_path, tt_out_path)

    # Spacy Tagger training
    if spacytagger:
        print("training Spacy Tagger")

        spacy_destdir = newdir + '/Spacy' + name
        spacy_outpath = Path(out_path + "/spacyTemp")
        cfgpath = Path("../SpacyTagger/config.cfg")
        trainpath = out_path + spacy_traindir
        devpath = out_path + spacy_devdir
        tempdirs.append(spacy_outpath)

        train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

        if bider:
            spacy_destdir = newdir + '/Spacy' + name + '_right'
            spacy_outpath = Path(out_path + "/spacyRTemp")
            trainpath = out_path + spacyR_traindir
            devpath = out_path + spacyR_devdir
            tempdirs.append(out_path + '/spacyRTemp/')

            train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

    # if stanzatagger:
        # print("training Stanza tagger")
        # destdir = newdir + '/Stanza' + name
        # if not os.path.isdir(destdir):
        #     os.mkdir(destdir)
        # if not os.path.isdir(out_path + '/StanzaTemp'):
        #     os.mkdir(out_path + '/StanzaTemp')

        # train_stanza("", out_path + stanza_traindir, out_path + stanza_devdir, out_path+'/StanzaTemp', "")
        # tempdirs.append(out_path + '/StanzaTemp')
        # copy_tree(out_path + '/StanzaTemp/model-best', destdir)

    return out_path + "/tune" + name


def train_super(path, trainfile, name="default", matrix="", taggers_array=None,):

    global tempfiles
    global tempdirs

    if matrix is "":
        if taggers_array is None:
            taggers_arr = os.listdir(path)
            taggers_array = ([])

            for t in taggers_arr:
                if '.par' in t or 'Spacy' in t or t.endswith('sr'):
                    taggers_array.append(path + '/' + t)

        matrices = ([])
        tagaccus = ([])
        tagsets = ([])

        with open(trainfile, 'r', encoding='utf-8') as f:
            tagslines = f.readlines()  # all lines including the blank ones
            tags = list(line.split('\t')[1] for line in tagslines if line != '\n')
            words = list(line.split('\t')[0] for line in tagslines)

        tag_freq = {i: tags.count(i) / len(tags) for i in set(tags)}

        targets = ([])
        if len(tagslines) < 75000:
            with open(path + '/prepared_ninety', 'w', encoding='utf8') as f:
                f.write('\n'.join(words))
            tempfiles.append(path + '/prepared_ninety')
            targets.append(path + '/prepared_ninety')

        else:
            alltext = '\n'.join(words)
            sents = alltext.split('\n\n\n')
            chunkn = round(len(words) / 50000)
            chunkovi = np.array_split(sents, chunkn)
            print(chunkn)

            for i, c in enumerate(chunkovi):
                with open(path + '/prepared' + str(i), 'w', encoding='utf-8') as temp:
                    temp.write('\n\n'.join(c))
                tempfiles.append(path + '/prepared' + str(i))
                targets.append(path + '/prepared' + str(i))

        for tagger in taggers_array:
            tlines = ([])
            for target in targets:
                tlines += tag_any(target, tagger, path)

            mat, accu, tagset, taggert = probtagToMatrix(tlines, tagger.split('/')[-1], tags)
            matrices.append(mat)
            tagaccus.append(accu)
            tagsets.append(tagset)

        flat_tagset = [item for sublist in tagsets for item in sublist]

        matricout = np.concatenate(matrices, axis=0)
        with open(path + "/matrix-prob.csv", 'w', encoding='utf-8') as m:
            m.write('result\t' + '\t'.join(flat_tagset) + '\n')
            for idx, line in enumerate(matricout.transpose()):
                # m.write(words[idx] + '\t') we can write the words but dont need to
                m.write(tags[idx] + '\t')
                np.savetxt(m, line[np.newaxis], delimiter='\t', fmt='%.4f')

        with open(path + "/tag_freq.csv", 'w', encoding='utf-8') as m:
            for tag in tag_freq:
                m.write(tag + '\t' + str(tag_freq[tag]) + '\n')

        with open(path + "/tag_accu.csv", 'w', encoding='utf-8') as m:
            for tagac in tagaccus:
                for tag in tagac:
                    m.write(tag + '\t' + str(tagac[tag]) + '\n')

        for tempf in tempfiles:
            if os.path.isfile(tempf):
                os.remove(tempf)
        for tempd in tempdirs:
            if os.path.exists(tempd):
                remove_tree(tempd)

        matrix = path + "/matrix-prob.csv"

    train_prob_net(matrix, path, name + ".pt")
