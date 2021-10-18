from tkinter import filedialog
from tkinter import *
import ntpath

from scripts.TagAll import tag_complex
from scripts.matrixworks import matrixworks
from scripts.lexmagic import lexmagic


def complex_test(tagger="", file_paths="", out_path="", modelname=""):

    accu = True
    distances = False

    tempfiles = ([])
    lexiconmagic = True
    transliterate = False
    Tk().withdraw()
    if file_paths == "":
        file_paths = filedialog.askopenfilenames(initialdir="./data",
                                                 title="Select text files",
                                                 filetypes=(
                                                     ("tagged files", "*lm *.tt .tag .txt .vrt .vert"),
                                                     ("all files", "*.*")))

    if accu:

        if tagger == "":
            tagger = filedialog.askdirectory(initialdir="./data/output", title="Select complex tagger directory")

            modelnames = filedialog.askopenfilenames(initialdir=tagger,
                                                   title="Select model or skip for default (net.prob.pt)",
                                                   filetypes=(("pt files", "*.pt"), ("all files", "*.*")))

    if out_path == "":
        out_path = filedialog.askdirectory(initialdir="./data/output", title="Select output directory")

    if lexiconmagic:
        lex_path = filedialog.askopenfilename(initialdir="./data/lexicon",
                                              title="Select lexicon file",
                                              filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        with open(lex_path, 'r', encoding='utf-8') as lex:
            entriesfull = [wordx.split('\t')[0] for wordx in lex.readlines()]
            entries_c = [wordx for wordx in entriesfull if wordx[0].isupper()]
            entries_l = [wordx for wordx in entriesfull if not wordx[0].isupper()]
            entries_u = [wordx for wordx in entries_c if wordx.isupper()]
            entries_c += [wordx for wordx in entries_c if not wordx.isupper()]

    for file in file_paths:
        for modelname in modelnames:
            modelname = ntpath.basename(modelname)
            print(modelname)
            results = ([])
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line is not '\n':
                    lines[i] = line.split('\t')[0]
                    results.append(line.split('\t')[1].rstrip())

            lines, orliglineslm = lexmagic(set(entries_u), set(entries_c), set(entries_l), lines)

            with open(out_path + '/temp0', 'w', encoding='utf-8') as temp:
                temp.write('\n'.join(lines))
                tempfiles.append(out_path + '/temp0')

            # with open(tagger + '/tag_accu.csv', 'r') as p:
            # accus = p.readlines()

            tag_accus = {}

            # for a in accus:
            # tag_accus[a.split('\t')[0]] = float(a.split('\t')[1].rstrip('\n'))

            tags, tagger_tags, probs = tag_complex(tagger, "", [out_path + '/temp0'], out_path, False, transliterate,
                                                   False, False, False, results, False, True, [modelname])

            # with open(out_path + '/tags', 'w', encoding='utf-8') as temp:
            #   temp.write('\n'.join(tags))

            # with open(out_path + '/tags', 'r', encoding='utf-8') as temp:
            #   tags = temp.readlines()
            csv = out_path + "/matrix-prob_tag.csv"
            # for x in tags:
            #     matrixworks(csv, tag_accus, x, results, tagger_tags)

            matrixworks(csv, tag_accus, tags[modelname], results, tagger_tags, probs[modelname], lines)

    if distances:

        csv = filedialog.askopenfilename(initialdir="./data/output",
                                         title="Select CSV file with tagging resluts",
                                         filetypes=(
                                             ("tagged files", ".csv"),
                                             ("all files", "*.*")))

        tager_dist(csv, results)


complex_test()
