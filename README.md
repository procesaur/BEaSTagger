# Bidirectional, Expandable and Stacked Tagger --- BEaST

This package provides POS-tagging using a system composed of [treetagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) and [spaCy](https://spacy.io/).
Package also provides training of a new system provided the necessary files.

## Installation
###### Google Collab
        `!pip install git+https://github.com/procesaur/BEaSTagger.git`

###### Linux and Windows
        `pip install -e git+https://github.com/procesaur/BEaSTagger.git#egg=beast`


## Tagging

from beast.tag import tag

tag(src=[`Text you want to tag`])
or
tag(src=[`path/to/file/totag`, `path/to/second/file`])

however, if you are not in a directory where the module is located default paths will not work.
you will have to set the paths in your os accordingly.

tag(src=[`Text you want to tag`],
model=`path/to/beast/model/dir`,
out_path=`/dir/for/output/dir`,
lexicons_path=`/dir/where/lexicons/are/located`,
tt_path=`/dir/where/treetagger/binaries`)

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

note>
- you need treetagger binaries
- you need reading permissions on the file you are tagging 
- you need writing permissions on the output folder
- you need at least one lexicon named 'default' at lexicons_path

note>
   if you are using windows and are located in the module folder you can tag via running the tag.py file.
   you will be prompted for the file you want to tag and results will be written in default dir.

## Training

from beast.train import train

train(file_path=`path/to/training/file/`)

however, if you are not in a directory where the module is located default paths will not work.
you will have to set the paths in your os accordingly.

train(file_path=[`Text you want to tag`],
beast_dir=`path/to/future/beast/model/dir`,
out_path=`/dir/for/output/dir`,
lexicons_path=`/dir/where/lexicons/are/located`,
tt_path=`/dir/where/treetagger/binaries`)

comeplete list of params:

    :param file_path: string > path to file (or url) that will be used for training. File must be in tsv form with a
    header, with the first column being the word and the last lemma.
    Names for tagset in between will be fetched from header - default of NONE results in tkinter input
    :param out_path: string > path to dir where model dir will be created - defaults to ./data/output
    :param pretrained: bool > do not train standalone taggers, use tunelist instead - defaults in False
    Only use when they are alrady pre-trained and available in single directory, and tune sets are available.
    :param test_ratio: float > ratio of training testing cutoff - defaults in 1, no cutoff
    :param tune_ratio: float > ratio of training tuning cutoff - defaults in 0.9, meaning 0.1 for tuning
    :param lexiconmagic: bool > do lexicon magic on training set? - default True
    :param transliterate: bool > transliterate training set? - default False
    :param lexicons_path: string > path to dir where lexicons are located - defaults to ./data/lexicon/
    :param beast_dir: string > path to dir where model is or be outputted - defaults to ./data/output/newBEaST
    :param lex_paths: dict{} > mapping between models (tagsets) and lexicons to be used for treetagger training -
    defaults of {} result in using the lexicons from the lexicons_path dir
    :param oc_paths: dict{} > mapping between models (tagsets) and openclass files to be used for treetagger training -
    defaults of {} result in using the openclass files from the lexicons_path dir
    :param tunepaths: dict{} > mapping between models (tagsets) and tuning files. necessary if pretrained - no default
    :param testing: bool > do the testing? - requires test_ratio of less than 1 and defaults in False
    :param onlytesting: string > skip all training and use this path to find test set - defaults in "",
     requires beast_dir
    :param fulltest: bool > output complete report for each tagger - defaults in False - compact metrics
    :param epochs: int > number of epochs for training stacked classifier
    :param batch_size: int > batch size for training stacked classifier
    :param learning_rate: float > learning_rate for training stacked classifier
    :param confidence: float > confidence line for beast tagger
    :param transfer: bool > use transfer learning > defaults in false
    :return: this function outputs trained model onto said location - testing returns test results, otherwise no returns

note>
- you need treetagger binaries
- you need reading permissions on the file you are using for training
- you need writing permissions on the output folder and the new beast dir
- you need at least one lexicon named 'default' at lexicons_path

note>
   if you are using windows and are located in the module folder you can tag via running the train.py file.
   you will be prompted for the file you want to use for training and results will be written in default dir.
