from TagAll import tag_complex
from tkinter import filedialog
from tkinter import *


# input > array of filepaths, directory path (default data/output), "POS" || "UD" (default POS)
def tag(files, out="./data/output/", tagset="POS"):
    if tagset == "POS":
        lem_par_path = "./data/output/default/TreeTaggerPOS.par"
    else:
        lem_par_path = "./data/output/default/TreeTagger.par"

    par_path = "./data/output/default/"
    lex_path = "./data/output/default/recnik_full_POS_lat.txt"
    modelname = tset + ".pt"
    for f in files:
        tag_complex(par_path, lex_path, [f], out,
                    True, True, True, False, False, None, True, False, modelname, lem_par_path)


Tk().withdraw()

file_paths = filedialog.askopenfilenames(initialdir="./data", title="Select plain text file",
                                         filetypes=(("txt files", "*.tt .tag .vrt .txt .xml"), ("all files", "*.*")))

out_path = filedialog.askdirectory(initialdir="./data/output", title="Select output directory")
tset = "POS"

tag(file_paths, out_path, tset)



