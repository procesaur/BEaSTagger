from re import compile, UNICODE, IGNORECASE, match
from os import path


dir = path.dirname(path.abspath(__file__))
xml = r'</?[šđžćč\-_\w]+( [šđžćč\-_\w]+=["\'].*["\'])*/?>'


def read_abbrevs(file):
    abbrevs = {'B': [], 'N': [], 'S': []}
    for line in open(path.join(dir, file), encoding='utf8'):
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
        'tag': xml,
        'mail': r'[\w.-]+@\w+(?:[.-]\w+)+',
        'mention': r'@[a-zšđžčć0-9_]+',
        'hashtag': r'#\w+(?:[.-]\w+)*',
        'emoticon': emoticon,
        'word': word,
        'arrow': r'<[-]+|[-]+>',
        'dot': r'[.!?/]{2,}',
        'space': r'\s+',
        'other': r'(.)\1*',
        'additional': r'[0-9][G|D|XL]',
        'period': r'[1-9][0-9]{0,2}0[\-－﹣﹘―—–‒‑‐᠆־]ih',
        'initials': r'[A-ZŠĐŽĆČАБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ]\.',
        'order': (
        'abbrev', 'period', 'num', 'url', 'htmlesc', 'tag', 'mail', 'initials', 'mention', 'hashtag', 'additional',
        'emoticon', 'word', 'arrow', 'dot', 'space', 'other')
    },

}

# transform abbreviation lists to sets for lookup during sentence splitting
for lang in abbrevs:
    for type in abbrevs[lang]:
        abbrevs[lang][type] = list(set([e.replace('\\.', '.') for e in abbrevs[lang][type]]))

spaces_re = compile(r'\s+', UNICODE)


def generate_tokenizer(lang):
    token_re = compile(r'|'.join([langs[lang][e] for e in langs[lang]['order']]), UNICODE | IGNORECASE)
    return token_re


def tokenize(tokenizer, paragraph):
    return [(e.group(0), e.start(0), e.end(0)) for e in
            tokenizer.finditer(paragraph.strip())]  # spaces_re.sub(' ',paragraph.strip()))]


def sentence_split(tokens, lang="sr"):
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


def tokenize_sentences(sentences, keepspace=False):
    output = []
    for sent in sentences:
        for token, start, end in sent:
            if not token.isspace() or keepspace:
                output.append(token)
        output.append("")
    return output


def sr_tokenize(text, keepspace=False):

    tokenizer = generate_tokenizer('sr')
    text = text.rstrip()
    text = text.replace("<", "\n<").replace(">", ">\n")
    text = text.split("\n")
    tokens = []

    for line in text:
        if line.strip() == '':
            continue
        elif match(xml, line):
            tokens.append(line)
            continue

        sentences = sentence_split(tokenize(tokenizer, line), 'sr')
        tokens.extend(tokenize_sentences(sentences, keepspace))

    return tokens


def gpt_tokenize(text):
    if "$$" in text:
        new = text.split("$$")
    else:
        tokens = sr_tokenize(text, keepspace=True)
        new = []
        last = ""
        for token in tokens:
            if token != " ":
                if last == " ":
                    new.append(last+token)
                else:
                    new.append(token)
            last = token
    return new


tokenizer = generate_tokenizer('sr')


def sentencize(text):
    res = []
    sentences = sentence_split(tokenize(tokenizer, text), 'sr')
    for s in sentences:
        xs = [x[0] for x in s]
        res.append("".join(xs))
    return res
