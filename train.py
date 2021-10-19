from scripts.TrainAll import train_taggers, train_super
from tkinter import Tk, filedialog as fd


# hide tkinterface
Tk().withdraw()
# use lexicon to enhance training (only use if you will use it in tagging)
lexiconmagic = True
split = True
ratio = 0.9
pretrained = True

if not pretrained:
    # select files that will be used for training >>>
    # training data (*format *WORD \t POS1 \t POS2 \t ... LEMMA\n )
    file_path = fd.askopenfilename(initialdir="./data/training", title="Select tagged text files",
                                           filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"),
                                                      ("all files", "*.*")))

    # output directory select
    out_path = fd.askdirectory(initialdir="./data/output",
                                       title="Select par output directory")

    tagsets = {}
    trainpaths = {}
    lex_paths = {}


    newdir = "ZTagger"

    with open(file_path, 'r', encoding='utf-8') as fl:
        lines = fl.readlines()  # all lines including the blank ones

    meta = lines[0].rstrip().split("\t")
    colsn = len(meta)
    del lines[0]

    newline = ""
    for i in range(1, colsn):
        newline += "\t"
    newline += "\n"

    lemacol = -1
    for i, m in enumerate(meta):
        if i > 0:
            if m != "lemma" and m != "lema":
                tagsets[m] = i
            else:
                lemacol = i

    for i, line in enumerate(lines):
        if line in ['\n', '', '\0', newline]:
            lines[i] = '\n'

     # shuffle dataset and split 9:1
    if split:

        count_sen = 0
        # remove new lines form the end of each line, because they are an array now
        for i, line in enumerate(lines):
            lines[i] = line.rstrip('\n')
            # check if line is empty, increase, sentence counter
            if lines[i] == "":
                count_sen += 1

        # check if text is split to sentences (is there count_sen), if not try to split by SENT tag.
        if count_sen < 1:
            # now add newline after each SENT, we are trying to break text into sentences
            for i, line in enumerate(lines):
                if 'SENT' in line:
                    lines[i] = lines[i] + '\n'
        # create sentence array
        text = re.sub(r'\n\n+', '\n\n', '\n'.join(lines)).strip()
        sentences = text.split('\n\n')
        # we set chunk sizes to X% of text - this will be used for training
        chunksize = len(sentences) * ratio

        # strip any extra newlines
        for i, sent in enumerate(sentences):
            if sent.endswith('\n'):
                sentences[i] = sent.rstrip('\n')

        # randomly shuffle sentences to get unbiased corpora
        random.shuffle(sentences)
        # initialize are chunks
        ninety = ([])
        ten = ([])
        # add first 90% of sentences to "ninety" array, and the rest to "ten" array
        for i, sent in enumerate(sentences):
            if i < chunksize:
                ninety.append(sent + "\n\n")
            else:
                ten.append(sent + "\n\n")
        # now read the lines from the "ninety" set, we will use this for training from now on
        lines = ''.join(ninety).rstrip('\n').split('\n')

        with open(out_path + "/testing _10", 'w', encoding='utf8') as tf:
            tf.write(''.join(ten).rstrip('\n'))

    for tagset in tagsets.keys():
        lex_paths[tagset] = filedialog.askopenfilename(initialdir="./data/lexicon",
                                                       title="Select lexicon file for " + tagset,
                                                       filetypes=(("text files", "*.txt"), ("all files", "*.*")))

    for tagset in tagsets.keys():
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

        path, trainpath = train_taggers(xlines, out_path, lex_paths[tagset], lexiconmagic, "_" + tagset, newdir)
        trainpaths[tagset] = trainpath

    for tagset in tagsets.keys():
        train_super(path, trainpaths[tagset], "", tagset)

else:
    # output directory select
    out_path = filedialog.askdirectory(initialdir="./data/output",
                                       title="Select tagger directory")
    tagsets = ("UD", "POS")
    for tagset in tagsets:
        file_path = filedialog.askopenfilename(initialdir="./data/training",
                                               title="Select tagged text files for " + tagset,
                                               filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"),
                                                          ("all files", "*.*")))
    for tagset in tagsets:
        train_super(out_path, file_path, "", tagset)