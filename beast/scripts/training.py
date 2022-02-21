import os
import numpy as np
from pathlib import Path
from distutils.dir_util import remove_tree
from os import path

from beast.scripts.pipeline import prepare_spacy, lexentries, makeconllu, ratio_split, write_chunks, lexmagic, probtagToMatrix
from beast.scripts.pipeline import prepare_stanza
from beast.scripts.conversion import convert as conv
from beast.scripts.torchworks import train_prob_net
from beast.scripts.tagging import tag_any
from beast.Classla.TrainClassla import train_stanza
from beast.SpacyTagger.spacyworks import train_spacy
from beast.SpacyTagger.spacyworks import gettagmap
from beast.TreeTagger.treetagger import train_treetagger


spacy_traindir = "/train.spacy"
spacy_devdir = "/dev.spacy"
spacyR_traindir = "/trainR.spacy"
spacyR_devdir = "/devR.spacy"
stanza_traindir = "/train.stanza"
stanza_devdir = "/dev.stanza"
stanzaR_traindir = "/trainR.stanza"
stanzaR_devdir = "/devR.stanza"

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
              '-lt 0.001',
              '-quiet']


def train_taggers(lines, out_path, lex_path, oc_path, name, newdir, tt_path, ratio=0.9, lexiconmagic=True,
                  transliterate=True, bidir=True, treetagger=False, spacytagger=False, stanzatagger=True, spacygpu=1):

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
        lines, orliglineslm, rc = lexmagic(set(entries_u), set(entries_c), set(entries_l), lines)
        # save lexicon magic

    train, tune = ratio_split(ratio, lines)

    # now read the lines from the "train" set, we will use this for training from now on
    lines = ''.join(train).rstrip('\n').split('\n')

    with open(out_path + "/tune" + name, 'w', encoding='utf8') as tf:
        tf.write(''.join(tune).rstrip('\n'))

    # Create inverted set > copy "lines" and reverse their order
    if bidir:
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
        if bidir:
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
    if treetagger and oc_path == "":
        with open(out_path + 'TreeTagger_openclass', 'w', encoding='utf8') as f:
            f.write('\n'.join(uniquepos))
        oc_path = out_path + 'TreeTagger_openclass'
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

        if stanzatagger:
            prepare_stanza(conlulines, tempdirs, out_path + stanza_traindir, out_path + stanza_devdir)
            tempfiles.append(out_path + stanza_traindir)
            tempfiles.append(out_path + stanza_devdir)

        if bidir:
            rconlulines = rlines.copy()
            # create conllu format
            rconlulines = makeconllu(rconlulines, tagmap)

            if spacytagger:
                prepare_spacy(rconlulines, tempdirs, out_path + spacyR_traindir, out_path + spacyR_devdir)
                tempfiles.append(out_path + spacyR_traindir)
                tempfiles.append(out_path + spacyR_devdir)

            if stanzatagger:
                prepare_stanza(rconlulines, tempdirs, out_path + stanzaR_traindir, out_path + stanzaR_devdir)
                tempfiles.append(out_path + stanzaR_traindir)
                tempfiles.append(out_path + stanzaR_devdir)

    # create output dirs on the disk
    if not os.path.isdir(newdir):
        os.mkdir(newdir)

    # TreeTagger training
    if treetagger:
        print("training TreeTagger")
        tt_in_path = out_path + '/TreeTagger_train'
        tt_out_path = newdir + '/TreeTagger' + name + '.par'
        train_treetagger(parameters, lex_path, oc_path, tt_in_path, tt_out_path, tt_path)

        if bidir:
            tt_in_path = out_path + '/TreeTaggerR_train'
            tt_out_path = newdir + '/TreeTagger' + name + '_right.par'
            train_treetagger(parameters, lex_path, oc_path, tt_in_path, tt_out_path, tt_path)

    # Spacy Tagger training
    if spacytagger:
        print("training Spacy Tagger")

        spacy_destdir = newdir + '/Spacy' + name
        spacy_outpath = Path(out_path + "/spacyTemp")
        cfgpath = path.join(path.dirname(__file__), "../SpacyTagger/config.cfg")
        trainpath = out_path + spacy_traindir
        devpath = out_path + spacy_devdir
        tempdirs.append(spacy_outpath)

        train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

        if bidir:
            spacy_destdir = newdir + '/Spacy' + name + '_right'
            spacy_outpath = Path(out_path + "/spacyRTemp")
            trainpath = out_path + spacyR_traindir
            devpath = out_path + spacyR_devdir
            tempdirs.append(out_path + '/spacyRTemp/')

            train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

    if stanzatagger:
        print("training Stanza tagger")
        destdir = newdir + '/Stanza' + name
        if not os.path.isdir(destdir):
            os.mkdir(destdir)
        if not os.path.isdir(out_path + '/StanzaTemp'):
            os.mkdir(out_path + '/StanzaTemp')

        pt = path.dirname(__file__) + "/../Classla/standard.pt"
        train_stanza(out_path + stanza_traindir, out_path + stanza_devdir, out_path+'/StanzaTemp', pt)
        tempdirs.append(out_path + '/StanzaTemp')
        # copy_tree(out_path + '/StanzaTemp/model-best', destdir)


def train_super(path, trainfile, tt_path, name="default", epochs=100, bs=32, lr=0.001,
                delete_tune=True, transfer_learn=False, matrix="", taggers_array=None):

    global tempfiles
    global tempdirs

    if delete_tune:
        tempfiles.append(trainfile)

    if matrix == "":
        if taggers_array is None:
            taggers_arr = os.listdir(path)
            taggers_array = ([])

            for t in taggers_arr:
                if '.par' in t or 'Spacy' in t or t.endswith('sr'):
                    taggers_array.append(path + '/' + t)

        if not transfer_learn:
            taggers_array = [ta for ta in taggers_array if "_" + name.lower() in ta.lower()]

        matrices = ([])
        tagaccus = ([])
        tagsets = ([])

        with open(trainfile, 'r', encoding='utf-8') as f:
            tagslines = f.readlines()  # all lines including the blank ones
            tags = list(line.split('\t')[1] for line in tagslines if line != '\n')
            words = list(line.split('\t')[0] for line in tagslines)

        tag_freq = {i: tags.count(i) / len(tags) for i in set(tags)}

        targets = write_chunks(words, path)

        for tagger in taggers_array:
            tlines = ([])
            for target in targets:
                tlines += tag_any(target, tagger, path, tt_path)

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

    train_prob_net(matrix, path, name + ".pt", epochs, bs, lr)
