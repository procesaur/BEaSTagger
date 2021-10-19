import os
from scripts.TrainAll import train_taggers, train_super
from scripts.pipeline import training_prep, ratio_split
from tkinter import Tk, filedialog as fd


def main(file_path="", out_path="./data/output/", pretrained=False, test_ratio=1, lex_paths={},
         lexicons_path="./data/lexicon/", tune_ratio=0.9, beast_dir="./data/output/newBEaST", tunepaths={}):

    # initiate lexicons
    lexicons = os.listdir(lexicons_path)
    lex_path = lexicons_path + "default"

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
                tf.write(''.join(test).rstrip('\n'))

        # prepare lexicon paths for treetagger if not supplied
        if lex_paths == {}:
            for tagset in tagsets.keys():
                for lexicon in lexicons:
                    if "_" + tagset.lower() in lexicon.lower():
                        lex_paths[tagset] = lexicons_path + lexicon
                if tagset not in lex_paths:
                    lex_paths[tagset] = lex_path

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
            tunepaths[tagset] = train_taggers(xlines, out_path, lex_paths[tagset], "_" + tagset, beast_dir, tune_ratio)

        for tagset in tagsets.keys():
            train_super(beast_dir, tunepaths[tagset], tagset)

    else:
        for tune in tunepaths:
            tunename = os.path.basename(tune)
            file_path = fd.askopenfilename(initialdir="./data/training",
                                           title="Select tagged text files for " + tunename,
                                           filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"),
                                                      ("all files", "*.*")))

            train_super(beast_dir, file_path, tunename)


if __name__ == "__main__":
    main()
