import os
import requests
from subprocess import DEVNULL, STDOUT, check_call


def isWindows():
    return os.name == 'nt'


def tag_treetagger(par_path, file_path, out_path, probability, lemmat, tt_path="./TreeTagger/bin"):
    # set system parameters for tagging
    args = ([])
    ext = ""
    if isWindows():
        ext = ".exe"

    args.append(tt_path + "tree-tagger" + ext)
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


def train_treetagger(params, lex_path, oc_path, in_path, out_path, tt_path="./TreeTagger/bin/"):
    args = ([])
    ext = ""
    if isWindows():
        ext = ".exe"

    if 'https://' in lex_path or 'http://' in lex_path:
        response = requests.get(lex_path)
        lex_path = response.text

    if 'https://' in oc_path or 'http://' in oc_path:
        response = requests.get(oc_path)
        oc_path = response.text

    args.append(tt_path + "train-tree-tagger" + ext)

    args.append(lex_path)
    args.append(oc_path)
    args.append(in_path)
    args.append(out_path)

    for param in params:
        if " " in param:
            args.append(param.split(' ')[0])
            args.append(param.split(' ')[1])
        else:
            args.append(param)

    with open(os.devnull, 'wb') as devnull:
        check_call(args, stdout=DEVNULL, stderr=STDOUT)
