from scripts.TagAll import tag_complex
import os
from tkinter import Tk, filedialog as fd


def main(files=None, model="./data/default", out_path="./data/output", lexicons_path="./data/lexicon/",
         transliterate=True, lexiconmagic=True, tokenize=True, MWU=False,
         onlyPOS=False, lemmat=True, lempos=False, modelnames=[], lemmatizers={}):

    # initiate lexicons
    lexicons = os.listdir(lexicons_path)
    lex_path = lexicons_path + "default"

    if not files:
        Tk().withdraw()
        files = fd.askopenfilenames(initialdir="./data/training", title="Select tagged text files",
                                    filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"), ("all files", "*.*")))

    if not modelnames:
        for m in os.listdir(model):
            if m.endswith(".pt"):
                modelnames.append(m)

    if lemmat and not lemmatizers:
        for modelx in modelnames:
            tagset = modelx.split('.')[0].lower()
            for lexicon in lexicons:
                if "_" + tagset in lexicon.lower():
                    lemmatizers[modelx] = lexicons_path + lexicon
            if modelx not in lemmatizers:
                lemmatizers[modelx] = lex_path

    for f in files:
        tag_complex(model, lex_path, [f], out_path, lexiconmagic, transliterate, tokenize, MWU, onlyPOS, None,
                    lemmat, False, modelnames, lemmatizers, lempos)


if __name__ == "__main__":
    main()