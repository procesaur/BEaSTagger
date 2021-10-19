import os
from subprocess import DEVNULL, STDOUT, check_call


def isWindows():
    return os.name == 'nt'


def tag_treetagger(par_path, file_path, out_path, probability, lemmat):
    # set system parameters for tagging
    args = ([])
    ext = ""
    if isWindows():
        ext = ".exe"

    args.append("./TreeTagger/bin/tree-tagger" + ext)
    args.append(par_path)
    args.append(file_path)

    if probability:
        args.append("-threshold")
        args.append("0.0001")
        args.append("-prob")

    if lemmat:
        args.append("-lemma")

    args.append("-token")
    args.append("-sgml")
    args.append("-no-unknown")

    args.append(out_path)

    with open(os.devnull, 'wb') as devnull:
        check_call(args, stdout=devnull, stderr=STDOUT)


def train_treetagger(params, lex_path, oc_path, in_path, out_path):
    args = ([])
    ext = ""
    if isWindows():
        ext = ".exe"

    args.append("./TreeTagger/bin/train-tree-tagger" + ext)
    args.append(lex_path)
    args.append(oc_path)
    args.append(in_path)
    args.append(out_path)

    for param in params:
        args.append(param)

    with open(os.devnull, 'wb') as devnull:
        check_call(args, stdout=devnull, stderr=STDOUT)
