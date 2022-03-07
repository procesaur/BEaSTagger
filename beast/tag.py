from beast.scripts.tagging import tag_complex
import os
from os import path
from tkinter import Tk, filedialog as fd


def tag(src=None, model="", out_path=".", lexicons_path="", transliterate=True, lexiconmagic=True,
        tokenize=True, MWU=False, stdout=False, quiet=True, onlyPOS=False, lemmat=True, lempos=False,
        modelnames=[], lemmafor=[], lemmatizers={}, probability=False, confidence=0.93):
    """
    :param src: string[] > list of files to be tagged (filepaths or urls) - default of NONE results in tkinter input
    :param model: string > path to BEaST model to use for tagging - defaults to ./data/default
    :param out_path: string > path to dir where tagged files will be written - defaults to ./data/output
    :param lexicons_path: string > path to dir where lexicons are located - defaults to ./data/lexicon/
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
    :param lemmafor: string[] > list of specific models (tagsets to be used) for lemmatization - default [] results in all available
    :param lemmatizers: dict{} > mapping between models (tagsets) and lexicons to be used for their lemmatization
    :param probability: bool > output probability > defaults in False
    :param confidence: float >
    :return: this function outputs tagged file onto said location - no returns
    """

    # initiate paths
    if model == "":
        model = path.join(path.dirname(__file__), "data/models/default")
    if lexicons_path == "":
        lexicons_path = path.join(path.dirname(__file__), "data/lexicon/")
    tt_path = path.join(path.dirname(__file__), "TreeTagger/bin/")

    # initiate lexicons
    lexicons = [x for x in os.listdir(lexicons_path) if "openclass" not in x]
    lex_path = lexicons_path + "default"

    if not src:
        Tk().withdraw()
        src = fd.askopenfilenames(initialdir="./data/training", title="Select tagged text files",
                                    filetypes=(("tagged files", "*.tt .tag .txt .vrt .vert .lm"), ("all files", "*.*")))

    if not modelnames:
        for m in os.listdir(model):
            if m.endswith(".pt") and m!="standard.pt":
                modelnames.append(m)

    if not lemmafor:
        for m in modelnames:
            tagset = m.split('.')[0].lower()
            lemmafor.append(tagset)

    if lemmat and not lemmatizers:
        for modelx in lemmafor:
            tagset = modelx.split('.')[0].lower()
            for lexicon in lexicons:
                if "_" + tagset in lexicon.lower():
                    lemmatizers[modelx] = lexicons_path + lexicon
            if modelx not in lemmatizers:
                lemmatizers[modelx] = lex_path

    if isinstance(src, str):
        tag_complex(model, lex_path, [src], out_path, tt_path, lexiconmagic, transliterate, tokenize, MWU, onlyPOS,
                    lemmat, False, quiet, modelnames, lemmatizers, lempos, probability, stdout, confidence)
    else:
        for f in src:
            tag_complex(model, lex_path, [f], out_path, tt_path, lexiconmagic, transliterate, tokenize, MWU, onlyPOS,
                        lemmat, False, quiet, modelnames, lemmatizers, lempos, probability, stdout, confidence)


if __name__ == "__main__":
    tag()
