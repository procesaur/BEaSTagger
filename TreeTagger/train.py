from datetime import datetime
from tkinter import filedialog
from tkinter import *
import os
import numpy as np
from tqdm import tqdm

cl = 3
dtg = 0.1
ecw = 0.2
atg = 1.1
# samo za pos
sw = 24
# za puno obelezavanje
# sw = 32

lexiconmagic = True
lexiconmagicpath = ""
tokenize = False
MWU = False
bider = True

# file choosing
Tk().withdraw()
file_paths = filedialog.askopenfilenames(initialdir="./data", title="Select tagged text files",
                                         filetypes=(("tagged files", "*.tt .tag .txt"), ("all files", "*.*")))

lex_path = filedialog.askopenfilename(initialdir="./lexicon", title="Select lexicon file",
                                      filetypes=(("text files", "*.txt"), ("all files", "*.*")))

oc_path = filedialog.askopenfilename(initialdir="./lexicon", title="Select open class file (tags list)",
                                     filetypes=(("text files", "*.txt"), ("all files", "*.*")))

lexiconmagicpath = filedialog.askopenfilename(initialdir="./par", title="Select lm changed file (if exists!)",
                                              filetypes=(("text files", "*.lm"), ("all files", "*.*")))

out_path = filedialog.askdirectory(initialdir="./par", title="Select par output directory")

# some initialization
lines = ([])
entries = ([])
entriesfull = ([])
disamb = ([])
linesfg = ([])
filenames = ([])
rc = 0
lc = 0

# reading files into memmory
print('reading files...')
for file in file_paths:
    with open(file, 'r', encoding='utf-8') as f:
        linesfg += f.readlines()  # All lines including the blank ones
        lines = list(line for line in linesfg if line != '\n')
        filepn = os.path.basename(file)
        filename, file_extension = os.path.splitext(filepn)
        filenames.append(filename)

filenamestring = '_'.join(filenames)
print(filenamestring)

# some lexicon magic (might take a while, based on file sizes)
if lexiconmagic:
    if lexiconmagicpath == "":
        with open(lex_path, 'r', encoding='utf-8') as lex:
            print('doing lexicon magic...  (might take a while, based on file sizes)')
            entriesfull += [wordx for wordx in lex.readlines()]
            entries += [wordx.split('\t')[0] for wordx in entriesfull]

        for i, line in tqdm(enumerate(lines)):
            lsplit = line.split('\t')
            word = lsplit[0]
            opos = lsplit[1]
            olema = lsplit[2]

            if word[0].isupper() and opos != 'ABB':
                wordlow = word.lower()
                wordcap = wordlow.capitalize()
                if wordcap in entries:
                    if wordcap != word:
                        lines[i] = wordcap + "\t" + opos + "\t" + olema
                        rc += 1
                else:
                    if wordlow in entries:
                        lines[i] = wordlow + "\t" + opos + "\t" + olema
                        rc += 1

        print('word replacements: ' + str(rc) + ' lines replacments: ' + str(lc))
        # save lexicon magic
        with open(out_path + "/" + filenamestring + ".lm", 'w', encoding='utf-8') as g:
            g.write(''.join(lines))

    else:
        with open(lexiconmagicpath, 'r', encoding='utf-8') as f:
            linesfg += f.readlines()  # All lines including the blank ones
            lines = list(line for line in linesfg if line != '\n')

# training parameters > can be readjusted
quiet = '-quiet'
print('training...')

with open(out_path + '/train', 'w', encoding='utf-8') as tr:
    tr.write(''.join(lines))

trainpaths = 'train-tree-tagger ' + ' "' + lex_path + '" "' + oc_path + '" "' \
             + out_path + '/train" "' + out_path + '/trained.par" '
parametres = '-cl ' + str(cl) + ' -dtg ' + str(dtg) + ' -ecw ' + str(ecw) + ' -atg ' \
             + str(atg) + ' -sw ' + str(sw) + ' -lt 0.001 ' + quiet

myCmd = trainpaths + parametres
# print(myCmd)
os.system(myCmd)

if bider:
    rlines = lines.copy()
    rlines.reverse()
    with open(out_path + '/train', 'w', encoding='utf-8') as tr:
        tr.write(''.join(rlines))
    trainpaths = 'train-tree-tagger ' + ' "' + lex_path + '" "' + oc_path + '" "' + out_path + '/train" "' \
                 + out_path + '/trained-right.par" '
    myCmd = trainpaths + parametres
    # print(myCmd)
    os.system(myCmd)

os.remove(out_path + "/train")
