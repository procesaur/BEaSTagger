import os
import numpy as np
from pathlib import Path
from os import path
from distutils.dir_util import copy_tree, remove_tree
from tkinter import Tk, filedialog as fd
import json
from shutil import copy2

from beast.scripts.pipeline import lexentries, makeconllu, ratio_split, write_chunks, lexmagic, probtagToMatrix
from beast.scripts.conversion import convert as conv
from beast.scripts.torchworks import train_prob_net
from beast.scripts.tagging import tag_any
from beast.StanzaTagger.train_stanza import train_stanza
from beast.StanzaTagger.stanzaworks import prepare_stanza
from beast.SpacyTagger.spacyworks import train_spacy, gettagmap, prepare_spacy
from beast.TreeTagger.treetagger import train_treetagger


resources = {"tokenize": {},
             "pos": {"standard": {"dependencies": [{"model": "pretrain", "package": "standard"}]}},
             "pretrain": {"standard": {}}, "default_processors": {"tokenize": "standard", "pos": "standard"},
             "default_dependencies": {"pos": [{"model": "pretrain", "package": "standard"}]}}

spacy_traindir = "/train.spacy"
spacy_devdir = "/dev.spacy"
spacyR_traindir = "/trainR.spacy"
spacyR_devdir = "/devR.spacy"
stanza_traindir = "/train.stanza"
stanza_devdir = "/dev.stanza"
stanza_goldir = "/gold.stanza"
stanzaR_traindir = "/trainR.stanza"
stanzaR_devdir = "/devR.stanza"
stanzaR_goldir = "/goldR.stanza"

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

contversion_def = 'CYRtoLAT'


def train_taggers(lines, out_path, lex_path, oc_path, name, newdir, tt_path, lexiconmagic, transliterate, ratio,
                  bidir, treetagger, spacytagger, stanzatagger, shorthand, stanzadp):

    global tempfiles
    global tempdirs

    # transliteration
    if transliterate:
        print('transliteration...')
        for i, line in enumerate(lines):
            lines[i] = conv(line, contversion_def)

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

        # transfer lines into conllu format that is usable.
        conlulines = makeconllu(lines, tagmap, stanzadp)

        if spacytagger:
            prepare_spacy(conlulines, tempdirs, out_path + spacy_traindir, out_path + spacy_devdir)
            tempfiles.append(out_path + spacy_traindir)
            tempfiles.append(out_path + spacy_devdir)

        if stanzatagger:
            pt = path.dirname(__file__) + "/../StanzaTagger/set.pt"
            if not os.path.isfile(pt):
                pt = fd.askopenfilename(initialdir="./data/training", title="Select pretrained vectors file",
                                        filetypes=(("tagged files", "*.pt"), ("all files", "*.*")))

            if stanzadp:
                dpt = path.dirname(__file__) + "/../StanzaTagger/dep.pt"
                if not os.path.isfile(dpt):
                    dpt = fd.askopenfilename(initialdir="./data/training", title="Select pretrained dependency parser",
                                             filetypes=(("tagged files", "*.pt"), ("all files", "*.*")))

            else:
                dpt = ""

            prepare_stanza(conlulines, tempfiles, out_path, stanza_traindir, stanza_devdir, dpt, pt)
            tempfiles.append(out_path + stanza_traindir)
            tempfiles.append(out_path + stanza_devdir)

        if bidir:
            # create conllu format
            rconlulines = makeconllu(rlines, tagmap, stanzadp)

            if spacytagger:
                prepare_spacy(rconlulines, tempdirs, out_path + spacyR_traindir, out_path + spacyR_devdir)
                tempfiles.append(out_path + spacyR_traindir)
                tempfiles.append(out_path + spacyR_devdir)

            if stanzatagger:
                prepare_stanza(rconlulines, tempfiles, out_path, stanzaR_traindir, stanzaR_devdir, dpt, pt)
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

        spacy_destdir = newdir + '/spacy' + name
        spacy_outpath = Path(out_path + "/spacyTemp")
        cfgpath = path.join(path.dirname(__file__), "../SpacyTagger/config.cfg")
        trainpath = out_path + spacy_traindir
        devpath = out_path + spacy_devdir
        tempdirs.append(spacy_outpath)

        train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

        if bidir:
            spacy_destdir = newdir + '/spacy' + name + '_right'
            spacy_outpath = Path(out_path + "/spacyRTemp")
            trainpath = out_path + spacyR_traindir
            devpath = out_path + spacyR_devdir
            tempdirs.append(out_path + '/spacyRTemp/')

            train_spacy(cfgpath, trainpath, devpath, spacy_outpath, spacy_destdir)

    if stanzatagger:
        print("training Stanza tagger")
        newjson = {}
        stanza_destdir = newdir + '/stanza' + name
        newjson[os.path.basename(stanza_destdir)] = resources
        stanza_outpath = out_path + "/stanzaTemp"
        train_stanza(out_path + stanza_traindir,
                     out_path + stanza_devdir,
                     stanza_outpath,
                     out_path + stanza_goldir,
                     shorthand, pt)
        tempdirs.append(stanza_outpath)
        copy_tree(stanza_outpath, stanza_destdir)

        if bidir:
            stanza_destdir = newdir + '/stanza' + name + '_right'
            stanza_outpath = out_path + "/stanzaRTemp"
            newjson[os.path.basename(stanza_destdir)] = resources
            train_stanza(out_path + stanzaR_traindir,
                         out_path + stanzaR_devdir,
                         stanza_outpath,
                         out_path + stanzaR_goldir,
                         shorthand, pt)
            tempdirs.append(stanza_outpath)
            copy_tree(stanza_outpath, stanza_destdir)

        respath = newdir + "/resources.json"
        if os.path.isfile(respath):
            with open(respath, "r", encoding="utf8") as jf1:
                oldjson = json.load(jf1)
            newjson.update(oldjson)

        with open(respath, "w", encoding="utf8") as jf2:
            json.dump(newjson, jf2)

        ptpath = newdir + "/" + os.path.basename(pt)
        if not os.path.isfile(ptpath):
            copy2(pt, newdir+"/standard.pt")


def train_super(path, trainfile, tt_path, name="default", epochs=100, bs=32, lr=0.001,
                delete_tune=True, transfer_learn=False, matrix="", taggers_array=None):

    global tempfiles
    global tempdirs

    if delete_tune:
        tempfiles.append(trainfile)

    if matrix == "":
        if taggers_array is None:
            taggers_arr = os.listdir(path)
            taggers_array = []

            for t in taggers_arr:
                if '.par' in t or 'spacy' in t or 'stanza' in t:
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
        for t in targets:
            tempfiles.append(path + "/" + t)

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
