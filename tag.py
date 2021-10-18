from scripts.TagAll import tag_complex
import os
from tkinter import Tk, filedialog as fd


def main(files=None, model="./data/default", out_path="./data/output", lex_path="./data/lexicon/default",
         transliterate=True, lexiconmagic=True, tokenize=True, MWU=False,
         onlyPOS=False, lemmat=True, lempos=False, modelnames=None, lemmatizers={}):

    if modelnames is None:
        modelnames = []
    if not files:
        Tk().withdraw()
        files = fd.askopenfilenames(initialdir="./data/training", title="Select tagged text files",
                                    filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"), ("all files", "*.*")))

    if not modelnames:
        for m in os.listdir(model):
            if m.endswith(".pt"):
                modelnames.append(m)

    if lemmat and not lemmatizers:
        for model in modelnames:
            lemmatizers[model] = lex_path

    for f in files:
        tag_complex(model, lex_path, [f], out_path, lexiconmagic, transliterate, tokenize, MWU, onlyPOS, None,
                    lemmat, False, modelnames, lemmatizers, lempos)


if __name__ == "__main__":
    main()

