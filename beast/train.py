import os
from os import path

from beast.scripts.training import train_taggers, train_super
from beast.scripts.testing import complex_test
from beast.scripts.pipeline import training_prep, ratio_split
from tkinter import Tk, filedialog as fd


def train(file_path="", out_path="", pretrained=False, test_ratio=0.9, tune_ratio=0.9,
          lexiconmagic=True, transliterate=False, lexicons_path="", beast_dir="",
          tt_path="", lex_paths={}, oc_paths={}, tunepaths={}, testing=False, onlytesting="",
          fulltest=False, epochs=120, batch_size=32, learning_rate=0.001, confidence=0.92, transfer=False):

    """
    :param file_path: string > path to file (or url) that will be used for training. File must be in tsv form with a
    header, with the first column being the word and the last lemma.
    Names for tagset in between will be fetched from header - default of NONE results in tkinter input
    :param out_path: string > path to dir where model dir will be created - defaults to current dir
    :param pretrained: bool > do not train standalone taggers, use tunelist instead - defaults in False
    Only use when they are alrady pre-trained and available in single directory, and tune sets are available.
    :param test_ratio: float > ratio of training testing cutoff - defaults in 1, no cutoff
    :param tune_ratio: float > ratio of training tuning cutoff - defaults in 0.9, meaning 0.1 for tuning
    :param lexiconmagic: bool > do lexicon magic on training set? - default True
    :param transliterate: bool > transliterate training set? - default False
    :param lexicons_path: string > path to dir where lexicons are located - defaults to ./data/lexicon/
    :param beast_dir: string > path to dir where model is or be outputted - defaults to ./data/output/newBEaST
    :param lex_paths: dict{} > mapping between models (tagsets) and lexicons to be used for treetagger training -
    defaults of {} result in using the lexicons from the lexicons_path dir
    :param oc_paths: dict{} > mapping between models (tagsets) and openclass files to be used for treetagger training -
    defaults of {} result in using the openclass files from the lexicons_path dir
    :param tunepaths: dict{} > mapping between models (tagsets) and tuning files. necessary if pretrained - no default
    :param testing: bool > do the testing? - requires test_ratio of less than 1 and defaults in False
    :param onlytesting: string > skip all training and use this path to find test set - defaults in "",
     requires beast_dir
    :param fulltest: bool > output complete report for each tagger - defaults in False - compact metrics
    :param epochs: int > number of epochs for training stacked classifier
    :param batch_size: int > batch size for training stacked classifier
    :param learning_rate: float > learning_rate for training stacked classifier
    :param confidence: float > confidence line for beast tagger
    :param transfer: bool > use transfer learning > defaults in false
    :return: this function outputs trained model onto said location - testing returns test results, otherwise no returns
    """

    #initiate paths
    if lexicons_path == "":
        lexicons_path = path.join(path.dirname(__file__), "lexicons_path")
    if beast_dir == "":
        beast_dir = path.join(path.dirname(__file__), "data/output/newBEaST")
    if tt_path == "":
        tt_path = path.join(path.dirname(__file__), "TreeTagger/bin/")
    

    # initiate lexicons
    lexicons = [x for x in os.listdir(lexicons_path) if "openclass" not in x]
    openclass = [x for x in os.listdir(lexicons_path) if "openclass" in x]

    # testing?
    if test_ratio >= 1:
        testing = False

    if onlytesting == "":
        if not pretrained:
            if file_path == "":
                Tk().withdraw()
                # select files that will be used for training >>>
                # training data (*format *WORD \t POS1 \t POS2 \t ... LEMMA\n )
                file_path = fd.askopenfilename(initialdir="./data/training", title="Select tagged text files",
                                               filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"),
                                                          ("all files", "*.*")))

            # prepare necessities from the file
            lines, lemacol, tagsets, newline, colsn = training_prep(file_path)

            # save the portion of dataset for testing?
            if test_ratio < 1:
                train, test = ratio_split(test_ratio, lines)
                lines = ''.join(train).rstrip('\n').split('\n')
                with open(out_path + "/testing", 'w', encoding='utf8') as tf:
                    tf.write("token\t")
                    for t in tagsets:
                        tf.write(t+"\t")
                    tf.write("lemma\n")
                    tf.write(''.join(test).rstrip('\n'))

            # prepare lexicon paths for treetagger if not supplied
            if lex_paths == {}:
                for tagset in tagsets.keys():
                    for lexicon in lexicons:
                        if "_" + tagset.lower() in lexicon.lower():
                            lex_paths[tagset] = lexicons_path + lexicon
                    if tagset not in lex_paths:
                        lex_paths[tagset] = lexicons_path + "default"

            # prepare openclass paths for treetagger if not supplied
            if oc_paths == {}:
                for tagset in tagsets.keys():
                    for oc in openclass:
                        if "_" + tagset.lower() in oc.lower():
                            oc_paths[tagset] = lexicons_path + oc
                    if tagset not in oc_paths:
                        oc_paths[tagset] = lexicons_path + "openclass"

            # train for each tagset
            for tagset in tagsets.keys():
                # prepare lines
                xlines = ([])
                c = tagsets[tagset]

                for line in lines:
                    if line in ['\n', '', '\0', newline]:
                        xlines.append('\n')
                    else:
                        if "\t" in line:
                            parts = line.rstrip().split("\t")
                            if lemacol > -1:
                                lem = "\t" + parts[lemacol]
                            else:
                                lem = ""
                            if len(parts) == colsn:
                                xlines.append(parts[0] + "\t" + parts[c] + lem)

                # train
                train_taggers(xlines, out_path, lex_paths[tagset], oc_paths[tagset], "_" + tagset, beast_dir, tt_path,
                              tune_ratio, lexiconmagic, transliterate)

            for tagset in tagsets.keys():
                train_super(beast_dir, out_path + "/tune_" + tagset, tt_path, tagset, epochs, batch_size, learning_rate)

        else:
            for tune in tunepaths:
                tunename = os.path.basename(tune)
                if tunepaths[tune] == "":
                    tunepaths[tune] = fd.askopenfilename(initialdir="./data/training",
                                                         title="Select tagged text files for " + tunename,
                                                         filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"),
                                                                    ("all files", "*.*")))

                train_super(beast_dir, tunepaths[tune], tt_path, tunename, epochs, batch_size, learning_rate, transfer)

        if testing:
            print("testing")
            complex_test(beast_dir, out_path + "/testing", lexiconmagic, transliterate, fulltest, confidence, out_path,
                         lexicons_path + "default", tt_path)

    else:
        print("testing")
        complex_test(beast_dir, onlytesting, lexiconmagic, transliterate, fulltest, confidence, out_path,
                     lexicons_path + "default", tt_path)


if __name__ == "__main__":
    train(testing=True, test_ratio=0.95, beast_dir="./data/output/newBEaST150", pretrained=True, tunepaths={"UD":"./data/output/tune_UD"})
