# Bidirectional, Expandable and Stacked Tagger --- BEaST

This package provides POS-tagging using a system composed of [treetagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/), [spaCy](https://spacy.io/) and [Stanza](https://stanfordnlp.github.io/) (again, as of version 1.1).
Package also provides training of a new system provided the necessary files.

If you use this package or generated data, please cite: https://doi.org/10.3390/app12105028

## Installation
###### Google Collab
        !pip install git+https://github.com/procesaur/BEaSTagger.git
        !chmod -R 775 '/usr/local/lib/python3.7/dist-packages/beast/TreeTagger/bin/'

###### Linux and Windows
        pip install -e git+https://github.com/procesaur/BEaSTagger.git#egg=beast
        [enable execution rights within beast/TreeTagger/bin/]


## Tagging

        from beast.tag import tag
and then:

        tag(src=`Text you want to tag`)
or

        tag(src=[`path/to/file/totag`, `path/to/second/file`])

however, if there is no model default located in the package you will have to specify the path to the one you want to use:
you can also download models from the web (zip file required). Pre-trained model for Serbian is available on GitHub @ https://github.com/procesaur/BEaSTagger/releases/download/Serbian/beast_serbian.zip. you can use it by download via
get_model(url, desiredname)

for example,

        get_model("https://github.com/procesaur/BEaSTagger/releases/download/Serbian/beast_serbian.zip", "default") 
will download the model and save it to beast/data/models/default, which is a defualt location for the the tagger and you can simple proceed with:

        tag(src=`Text you want to tag`)

        tag(src=`Text you want to tag`, model=`path/to/beast/model/dir`)

comeplete list of params:

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

note>
- you need reading permissions on the file you are tagging 
- you need writing permissions on the output folder
- you need at least one lexicon named 'default' at lexicons_path (this is provided for in the package - for Serbian)

note>
   if you are using windows and are located in the module folder you can tag via running the tag.py file.
   you will be prompted for the file you want to tag and results will be written in default dir.

## Training

        from beast.train import train
and then:

        train(file_path=`path/to/training/file/`)

or to specify the path to where you want to station your new model:

        train(file_path=[`Text you want to tag`], beast_dir=`path/to/future/beast/model/dir`,)

comeplete list of params:

    :param file_path: string > path to file (or url) that will be used for training. File must be in tsv form with a
    header, with the first column being the word and the last lemma.
    Names for tagset in between will be fetched from header - default of NONE results in tkinter input
    :param out_path: string > path to dir where model dir will be created - defaults to current dir
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
    :param transfer: bool > use transfer learning > defaults in False
    :param bidir: bool > bidireectional training > default sin True
    :param treetagger: bool > use TreeTagger for composition > defaults in True
    :param spacytagger: bool > use spaCy for composition > defaults in True
    :param stanzatagger: bool > use Stanza for composition > defaults in False
    :param shorthand: string > Stanza langauge code > defaults in the one for Serbian
    :param stanzadp: bool > use dependency parsing for stanza training: defualts in False, requires pretrained file
    :return: this function outputs trained model onto said location - testing returns test results, otherwise no returns

note>
- you need reading permissions on the file you are using for training
- you need writing permissions on the output folder and the new beast dir
- you need at least one lexicon named 'default' at lexicons_path

note>
   if you are using windows and are located in the module folder you can tag via running the train.py file.
   you will be prompted for the file you want to use for training and results will be written in default dir.
   
   
## Usecase using pretrained model (for Serbian) - google colab
   
        !pip install git+https://github.com/procesaur/BEaSTagger.git
        !chmod -R 775 '/usr/local/lib/python3.7/dist-packages/beast/TreeTagger/bin/'
        from beast.tag import tag
        from beast.other import get_model
        get_model("https://github.com/procesaur/BEaSTagger/releases/download/Serbian/beast_serbian.zip", "default")
        tag("Dobro veče, dobri moji ljudi!", model="newBEaST", stdout=True)
        
Dobrodošli	ADJ	dobrodošao	A	dobrodošao

,	PUNCT	,	PUNCT	,

dobri	ADJ	dobar	A	dobar

moji	DET	moj	PRO	moj

ljudi	NOUN	ljudi	N	ljudi

!	PUNCT	!	SENT	!
   
## Usecase without additional data (for Serbian) - google colab

        !pip install git+https://github.com/procesaur/BEaSTagger.git
        !chmod -R 775 '/usr/local/lib/python3.7/dist-packages/beast/TreeTagger/bin/'
        from beast.train import train
        from beast.tag import tag
        
        
        train(file_path="/usr/local/lib/python3.7/dist-packages/beast/data/training/SrpKor4Tagging", testing=True)
        
        
preparing...

ℹ Grouping every 10 sentences into a document.

ℹ Grouping every 10 sentences into a document.

training TreeTagger

training Spacy Tagger

...

...

tagger	w_prec	w_rec	w_f1	m_prec	m_rec	m_f1

Spacy_pos	0.866	0.866	0.865	0.896	0.709	0.727

TreeTagger_pos.par	0.937	0.939	0.936	0.912	0.788	0.804

Spacy_pos_right	0.873	0.871	0.871	0.849	0.687	0.701

TreeTagger_pos_right.par	0.928	0.93	0.927	0.891	0.781	0.788

high	0.971	0.971	0.97	0.947	0.84	0.855

jury	0.971	0.971	0.97	0.95	0.829	0.848

BEaST	0.973	0.973	0.972	0.944	0.866	0.87
                
                
        tag("Dobro veče, dobri moji ljudi!", model="newBEaST", stdout=True)
        
        
Dobrodošli	ADJ	dobrodošao	A	dobrodošao

,	PUNCT	,	PUNCT	,

dobri	ADJ	dobar	A	dobar

moji	DET	moj	PRO	moj

ljudi	NOUN	ljudi	N	ljudi

!	PUNCT	!	SENT	!


