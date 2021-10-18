import re
import sys
import os
from tqdm import tqdm


if sys.platform == 'win32':
    from signal import signal, SIG_DFL
else:
    from signal import signal, SIGPIPE, SIG_DFL

if sys.platform != 'win32':
    signal(SIGPIPE, SIG_DFL)

reldir = os.path.dirname(os.path.abspath(__file__))


def read_abbrevs(file):
    abbrevs = {'B': [], 'N': [], 'S': []}
    for line in open(os.path.join(reldir, file), encoding='utf8'):
        if not line.startswith('#'):
            abbrev, type = line.strip().split('\t')[:2]
            abbrevs[type].append(abbrev)
    return abbrevs


abbrevs = {
    'sr': read_abbrevs('sr.abbrev'),
}

num = r'(?:(?<!\d)[+-])?\d+(?:[.,:/]\d+)*(?:[.](?!\.)|-[^\W\d_]+)?'
# emoswithspaces emoticon=r'[=:;8][\'-]*(?:\s?\)+|\s?\(+|\s?\]+|\s?\[
# +|\sd\b|\sp\b|d+\b|p+\b|s+\b|o+\b|/|\\|\$|\*+)|-\.-|\^_\^|\([\W]+\)|<3|</3|<\\3|\\o/'
emoticon = r'[=:;8][\'-]*(?:\)+|\(+|\]+|\[+|d\b|p\b|d+\b|p+\b|s+\b|o+\b|/|\\|\$|\*+)|-\.-|\^_\^|\([' \
           r'^\w\s]+\)|<3|</3|<\\3|\\o/ '
word = r'(?:[*]{2,})?\w+(?:[@­\'-]\w+|[*]+\w+)*(?:[*]{2,})?'

langs = {

    'sr': {
        'abbrev': r'|'.join(abbrevs['sr']['B'] + abbrevs['sr']['N'] + abbrevs['sr']['S']),
        'num': num,
        'url': r'https?://[-\w/%]+(?:[.#?=&@;][-\w/%]+)+|\b\w+\.(?:\w+\.)?('
               r'?:com|org|net|gov|edu|int|io|eu|si|hr|rs|ba|me|mk|it|at|hu|bg|ro|al|de|ch|be|dk|se|no|es|pt|ie|fr|fi'
               r'|cl|co|bo|br|gr|ru|uk|us|by|cz|sk|pl|lt|lv|lu|ca|in|tr|il|iq|ir|hk|cn|jp|au|nz)/?\b',
        'htmlesc': r'&#?[a-zšđžčć0-9]+;',
        'tag': r'</?[a-zšđžčć][\w:]*>|<[a-zšđžčć][\w:]*/?>',
        'mail': r'[\w.-]+@\w+(?:[.-]\w+)+',
        'mention': r'@[a-zšđžčć0-9_]+',
        'hashtag': r'#\w+(?:[.-]\w+)*',
        'emoticon': emoticon,
        'word': word,
        'arrow': r'<[-]+|[-]+>',
        'dot': r'[.!?/]{2,}',
        'space': r'\s+',
        'other': r'(.)\1*',
        'order': (
        'abbrev', 'num', 'url', 'htmlesc', 'tag', 'mail', 'mention', 'hashtag', 'emoticon', 'word', 'arrow', 'dot',
        'space', 'other')
    },

}

# transform abbreviation lists to sets for lookup during sentence splitting
for lang in abbrevs:
    for type in abbrevs[lang]:
        abbrevs[lang][type] = set([e.replace('\\.', '.') for e in abbrevs[lang][type]])

spaces_re = re.compile(r'\s+', re.UNICODE)


def generate_tokenizer(lang):
    els = langs[lang]
    token_re = re.compile(r'|'.join([langs[lang][e] for e in langs[lang]['order']]), re.UNICODE | re.IGNORECASE)
    return token_re


def tokenize(tokenizer, paragraph):
    return [(e.group(0), e.start(0), e.end(0)) for e in
            tokenizer.finditer(paragraph.strip())]  # spaces_re.sub(' ',paragraph.strip()))]


def sentence_split_nonstd(tokens, lang):
    boundaries = [0]
    for index in range(len(tokens) - 1):
        token = tokens[index][0]
        if token[0] in '.!?…':  # if sentence ending punctuation
            boundaries.append(index + 1)
        elif token.endswith('.'):  # if abbreviation
            if token.lower() not in abbrevs[lang]['N']:  # if not in non-splitting abbreviationx
                if token.lower() in abbrevs[lang]['S']:  # if in splitting abbreviationx
                    boundaries.append(index + 1)
                elif len(token) > 2:
                    if tokens[index + 1][0][0].isupper():  # else if next is uppercase
                        boundaries.append(index + 1)
                        continue
                    if index + 2 < len(tokens):  # else if next is space and nextnext is uppercase
                        if tokens[index + 1][0][0].isspace() and tokens[index + 2][0][0].isupper():
                            # tokens[index+1][0][0] not in u'.!?…':
                            boundaries.append(index + 1)
    boundaries.append(len(tokens))
    sents = []
    for index in range(len(boundaries) - 1):
        sents.append(tokens[boundaries[index]:boundaries[index + 1]])
    return sents


def sentence_split(tokens, lang):
    boundaries = [0]
    for index in range(len(tokens) - 1):
        token = tokens[index][0]
        if token[0] in '.!?…' or (token.endswith('.') and token.lower() not in abbrevs[lang]['N'] and len(token) > 2 and
                                  tokens[index + 1][0][0] not in '.!?…'):
            if tokens[index + 1][0][0].isupper():
                boundaries.append(index + 1)
                continue
            if index + 2 < len(tokens):
                if tokens[index + 2][0][0].isupper():
                    if tokens[index + 1][0].isspace() or tokens[index + 1][0][0] in '-»"\'„':
                        boundaries.append(index + 1)
                        continue
            if index + 3 < len(tokens):
                if tokens[index + 3][0][0].isupper():
                    if tokens[index + 1][0].isspace() and tokens[index + 2][0][0] in '-»"\'„':
                        boundaries.append(index + 1)
                        continue
            if index + 4 < len(tokens):
                if tokens[index + 4][0][0].isupper():
                    if tokens[index + 1][0].isspace() and tokens[index + 2][0][0] in '-»"\'„' \
                            and tokens[index + 3][0][0] in '-»"\'„':
                        boundaries.append(index + 1)
                        continue
        if token[0] in '.!?…':
            if index + 2 < len(tokens):
                if tokens[index + 2][0][0].isdigit():
                    boundaries.append(index + 1)
                    continue
    boundaries.append(len(tokens))
    sents = []
    for index in range(len(boundaries) - 1):
        sents.append(tokens[boundaries[index]:boundaries[index + 1]])
    return sents


process = {'standard': lambda x, y, z: sentence_split(tokenize(x, y), z),
           'nonstandard': lambda x, y, z: sentence_split_nonstd(tokenize(x, y), z)}


def to_text(sent):
    text = ''
    for idx, (token, start, end) in enumerate(sent):
        if idx == 0 and token[0].isspace():
            continue
        text += token
    return text + '\n'


def represent_tomaz(input, par_id, conllu, bert, tagger):
    output = ''
    token_id = 0
    sent_id = 0
    if conllu:
        output += '# newpar id = ' + str(par_id) + '\n'
    for sent_idx, sent in enumerate(input):
        sent_id += 1
        token_id = 0
        if conllu:
            output += '# sent_id = ' + str(par_id) + '.' + str(sent_id) + '\n'
            output += '# text = ' + to_text(sent)
        for token_idx, (token, start, end) in enumerate(sent):
            if not token[0].isspace():
                token_id += 1
                if conllu:
                    SpaceAfter = True
                    if len(sent) > token_idx + 1:
                        SpaceAfter = sent[token_idx + 1][0].isspace()
                    elif len(input) > sent_idx + 1:
                        SpaceAfter = input[sent_idx + 1][0][0].isspace()
                    if SpaceAfter:
                        output += str(token_id) + '\t' + token + '\t_' * 8 + '\n'
                    else:
                        output += str(token_id) + '\t' + token + '\t_' * 7 + '\tSpaceAfter=No\n'
                elif bert:
                    output += token + ' '
                elif tagger:
                    output += token + '\n'
                else:
                    output += str(par_id) + '.' + str(sent_id) + '.' + str(token_id) + '.' + str(start + 1) + '-' + str(
                        end) + '\t' + token + '\n'
        if bert:
            output = output.strip()
        output += '\n'
    return output


def rel_tokenize(text, out_path, lang='sr', document=False, nonstandard=False, conllu=False, bert=False, tagger=True):

    def nextpar(text):
        for x in text:
            yield x

    if document:
        conllu = True
    mode = 'standard'
    if nonstandard:
        mode = 'nonstandard'
    tokenizer = generate_tokenizer(lang)
    par_id = 0

    newtext = ''
    parn = len(text)
    print(str(parn) + " paragraphs")
    for line in tqdm(nextpar(text), total=parn):
        if line.strip() == '':
            continue
        elif re.match(r"^.*<!--.*$|^.*-->.*$|^.*<.*>.*$", line):
            newtext += line + '\n'
            continue
        par_id += 1
        if document:
            if line.startswith('# newdoc id = '):
                par_id = 0
                sys.stdout.write(line)
                continue

        adds = represent_tomaz(process[mode](tokenizer, line, lang), par_id, conllu, bert, tagger)
        newtext += represent_tomaz(process[mode](tokenizer, line, lang), par_id, conllu, bert, tagger)
        if bert:
            newtext += '\n'

    with open(out_path + "/tempw", 'w', encoding='utf-8') as w:
        w.write(newtext)

    return newtext.split('\n')
