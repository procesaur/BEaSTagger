from scripts.tagging import tag_complex
import os
from tkinter import Tk, filedialog as fd


def tag(src=None, model="./data/default", out_path="./data/output", lexicons_path="./data/lexicon/",
        tt_path="./TreeTagger/bin/", transliterate=True, lexiconmagic=True, tokenize=True, MWU=False, stdout=False,
        quiet=True, onlyPOS=False, lemmat=True, lempos=False, modelnames=[], lemmatizers={}, probability=False,
        confidence=0.93):
    """
    :param src: string[] > list of files to be tagged (filepaths or urls) - default of NONE results in tkinter input
    :param model: string > path to BEaST model to use for tagging - defaults to ./data/default
    :param out_path: string > path to dir where tagged files will be written - defaults to ./data/output
    :param lexicons_path: string > path to dir where lexicons are located - defaults to ./data/lexicon/
    :param tt_path: string > path to treetagger folder where executables are located - defaults to ./TreeTagger/bin/
    :param transliterate: bool > transliterate text into latin? - defaults in True
    :param lexiconmagic: bool > adapt text to a lexicon for perhaps improved results? - defaults in True
    :param tokenize: bool > tokenize the text? - defaults in True, only use False if supplying tokenized text
    :param MWU: bool > don't tokenize MWU? - defaults in False, do not use True if not sure
    :param stdout bool > don't write to a file, but output in console - defaults in False
    :param quiet bool > don't print info to console > defaults in True
    :param onlyPOS: bool > strip tags of additional information (after :) - defaults in False
    :param lemmat: bool > lemmatize the text? > defaults in True
    :param lempos: bool > output lempos columns? > defaults in False
    :param modelnames: string[] > list of specific models (tagsets to be used) - default [] results in all available
    :param lemmatizers: dict{} > mapping between models (tagsets) and lexicons to be used for their lemmatization
    :param probability: bool > output probability > defaults in False
    :param confidence: float >
    :return: this function outputs tagged file onto said location - no returns
    """

    # initiate lexicons
    lexicons = [x for x in os.listdir(lexicons_path) if "openclass" not in x]
    lex_path = lexicons_path + "default"

    if not src:
        Tk().withdraw()
        src = fd.askopenfilenames(initialdir="./data/training", title="Select tagged text files",
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

    for f in src:
        tag_complex(model, lex_path, [f], out_path, tt_path, lexiconmagic, transliterate, tokenize, MWU, onlyPOS,
                    lemmat, False, quiet, modelnames, lemmatizers, lempos, probability, stdout, confidence)


if __name__ == "__main__":
    tag(probability=True, files=["Zdravo živo komšo."])
