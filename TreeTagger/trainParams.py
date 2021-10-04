from datetime import datetime
from tkinter import filedialog
from tkinter import *
import os
import numpy as np
from tqdm import tqdm

lexiconmagic = True
lexiconmagicpath = ""
tokenize = False
MWU = False

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
lines = ([])
entries = ([])
entriesfull = ([])
disamb = ([])
linesfg = ([])
filenames = ([])
# some initialization

rc = 0
lc = 0
with open(out_path + "/training-result.log", 'w+', encoding='utf-8') as g:
    g.write('cl\tdtg\tecw\tatg\tsw\tfull\tfpos\tpos\tlemma\n')

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
# quiet=''
quiet = '-quiet'
n = 10

clrange = range(3, 4)
dtgrange = np.arange(0.1, 0.2, 0.1)
ecwrange = np.arange(0.2, 0.3, 0.1)
atgrange = range(1, 2)
# swrange = range(16, 22, 6)
swrange = range(32, 33, 8)

print('training and testing...')
chunks = np.array_split(lines, n)
for cl in clrange:
    for dtg in dtgrange:
        for ecw in ecwrange:
            for atg in atgrange:
                for sw in swrange:

                    # some more initialization
                    acc = 0
                    accfpos = 0
                    accpos = 0
                    acclemma = 0

                    for i in range(0, n):

                        copy = chunks.copy()
                        del copy[i]
                        test = chunks[i]
                        train = np.concatenate(copy)

                        with open(out_path + '/test', 'w', encoding='utf-8') as tt:
                            tt.write(''.join(test))

                        with open(out_path + '/train', 'w', encoding='utf-8') as tr:
                            tr.write(''.join(train))

                        trainpaths = 'train-tree-tagger ' + ' "' + lex_path + '" "' + oc_path + '" "' + out_path + \
                                     '/train" "' + out_path + '/temp.par" '
                        parametres = '-cl ' + str(cl) + ' -dtg ' + str(dtg) + ' -ecw ' + str(ecw) + ' -atg ' + \
                                     str(atg) + ' -sw ' + str(sw) + ' -lt 0.001 ' + quiet

                        myCmd = trainpaths + parametres
                        # print(myCmd)
                        os.system(myCmd)

                        with open(out_path + '/test', 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            linesf = lines.copy()
                            for idx, line in enumerate(lines):
                                lines[idx] = line.split('\t')[0]

                        with open(out_path + "/test", 'w', encoding='utf-8') as g:
                            g.write('\n'.join(lines))

                        myCmd = 'tree-tagger "' + out_path + '/temp.par" "' + out_path + '/test" "' + out_path + \
                                '/temp" -token -lemma -sgml -no-unknown ' + quiet
                        # print(myCmd)
                        os.system(myCmd)

                        true = 0
                        truefpos = 0
                        truepos = 0
                        truelemma = 0
                        false = 0
                        falsefpos = 0
                        falsepos = 0
                        falselemma = 0
                        diff = ""

                        with open(out_path + '/temp', 'r', encoding='utf-8') as ft:
                            linest = ft.readlines()
                            for idx, line in enumerate(linest):
                                if "@card@" in linest[idx]:
                                    # if linest[idx].split('\t')[2] == "@card@":
                                    linest[idx] = linest[idx].replace("@card@", linest[idx].split('\t')[0])
                                # if linesf[idx].split('\t')[1] != 'SENT' and linesf[idx].split('\t')[1] != 'PUNCT'
                                # and linesf[idx].split('\t')[1] != 'NUM' and linesf[idx].split('\t')[1] != 'ABB':
                                if linesf[idx].split('\t')[1] != 'SENT' and linesf[idx].split('\t')[1] != 'PUNCT' and \
                                        linesf[idx].split('\t')[1] != 'ABB':
                                    yyt = linesf[idx]
                                    xxx = linest[idx]
                                    if linesf[idx] == linest[idx]:
                                        # if linesf[idx] == linest[idx] or linesf[idx].split('\t')[1] ==
                                        # 'SENT' or linesf[idx].split('\t')[1] == 'PUNCT':
                                        true += 1
                                        truefpos += 1
                                        truepos += 1
                                        truelemma += 1
                                    else:
                                        false += 1
                                        if linesf[idx].split('\t')[1] == linest[idx].split('\t')[1]:
                                            truefpos += 1
                                        else:
                                            falsefpos += 1
                                        if linesf[idx].split('\t')[2] == linest[idx].split('\t')[2]:
                                            truelemma += 1
                                        else:
                                            falselemma += 1
                                        if linesf[idx].split('\t')[1].split(':')[0] == \
                                                linest[idx].split('\t')[1].split(':')[0]:
                                            truepos += 1
                                        else:
                                            falsepos += 1

                        rate = 100 / (true + false) * true
                        ratefpos = 100 / (truefpos + falsefpos) * truefpos
                        ratepos = 100 / (truepos + falsepos) * truepos
                        ratelemma = 100 / (truelemma + falselemma) * truelemma
                        print("accuracy> " + str(rate) + "% true=" + str(true) + " false=" + str(false))

                        acc += rate
                        accfpos += ratefpos
                        accpos += ratepos
                        acclemma += ratelemma

                    acc = acc / n
                    accfpos = accfpos / n
                    accpos = accpos / n
                    acclemma = acclemma / n

                    print('-cl ' + str(cl) + ' -dtg ' + str(dtg) + ' -ecw ' + str(ecw) + ' -atg ' + str(
                        atg) + ' -sw ' + str(sw))
                    print(str(n) + '-fold average: full=' + str(acc) + '% fpos=' + str(accfpos) + '% pos=' + str(
                        accpos) + '% lemma=' + str(acclemma) + '%')

                    with open(out_path + "/training-result.log", 'a+', encoding='utf-8') as g:
                        g.write(
                            str(cl) + '\t' + str(dtg) + '\t' + str(ecw) + '\t' + str(atg) + '\t' + str(sw) + '\t' + str(
                                acc) + '\t' + str(accfpos) + '\t' + str(accpos) + '\t' + str(acclemma) + '\n')

os.remove(out_path + "/temp")
os.remove(out_path + "/test")
os.remove(out_path + "/train")
