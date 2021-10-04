from TagAll import tag_complex, tag_any
from tkinter import filedialog
from tkinter import *
import ntpath

# variables
transliterate = True
lexiconmagic = True
tokenize = True
MWU = False
probability = True
onlyPOS = False
lemmat = True
lempos = True
complext = True
par_path = ""
modelname = ""
lemmatizers = {}

Tk().withdraw()
file_paths = filedialog.askopenfilenames(initialdir="./data", title="Select plain text file",
                                         filetypes=(("txt files", "*.tt .tag .vrt .txt .xml"), ("all files", "*.*")))
if not complext:
    par_path = filedialog.askopenfilename(initialdir="./data", title="Select par file",
                                          filetypes=(("par files", "*.par"), ("all files", "*.*")))
if par_path == "":
    par_path = filedialog.askdirectory(initialdir="./data/output", title="Select tagger")

if complext:
    modelnames = list(filedialog.askopenfilenames(initialdir=par_path,
                                             title="Select model or skip for default (net.prob.pt)",
                                             filetypes=(("pt files", "*.pt"), ("all files", "*.*"))))

if modelnames == "":
    modelnames = ["net.prob.pt"]
for i, model in enumerate(modelnames):
    modelnames[i] = ntpath.basename(model)

out_path = filedialog.askdirectory(initialdir="./data/output", title="Select output directory")

if lexiconmagic:
    lex_path = filedialog.askopenfilename(initialdir="./data/lexicon", title="Select lexicon file (for lexical magic) ",
                                          filetypes=(("text files", "*.txt"), ("all files", "*.*")))
else:
    lex_path = ""

if lemmat:
    for model in modelnames:
        lem_par_path = filedialog.askopenfilename(initialdir="./data/lexicon",
                                                  title="Select lemma dictionary (or Treetagger par)  for " + model,
                                                  filetypes=(("par files", "*.par *.txt *.dic"), ("all files", "*.*")))
        if lem_par_path != "":
            lemmatizers[model] = lem_par_path


for f in file_paths:
    if complext:
        tag_complex(par_path, lex_path, [f], out_path, lexiconmagic,
                    transliterate, tokenize, MWU, onlyPOS, None, lemmat, False, modelnames, lemmatizers, lempos)
    else:
        tag_any(transliterate, lexiconmagic, tokenize, MWU, probability, onlyPOS, lemmat, [f], par_path, out_path, True,
                lex_path)
