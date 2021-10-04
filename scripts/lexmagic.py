from tqdm import tqdm

def lexmagic(entries_u, entries_c, entries_l, lines, synonyms=None):
    origlines = lines.copy()
    rc = 0
    # print('doing lexicon magic...  (might take a while, based on file sizes)')
    listexcept = ["„", "“", "”", "*"]
    linesx = ([])

    def l(ls):
        for x in ls:
            yield x

    # for each line take word pos and lemma
    for line in l(lines):
        lsplit = line.split('\t')
        if len(lsplit) > 2:
            opos = "\t" + lsplit[1]
            olema = "\t" + lsplit[2]
        elif len(lsplit) == 2:
            opos = "\t" + lsplit[1]
            olema = ""
        else:
            opos = ""
            olema = ""
        word = lsplit[0].rstrip('\n')

        if word != '':

            # if the first letter in a word is capitalized
            if word[0].isupper():
                wordlow = word.lower()  # generate lowercaps word
                wordcap = wordlow.capitalize()  # generate word with a capital

                if word.isupper():
                    if word in entries_u:
                        pass
                    elif wordcap in entries_c:
                        word = wordcap
                        rc += 1

                    elif wordlow in entries_l:
                        word = wordlow
                        rc += 1

                else:
                    if wordcap in entries_c:
                        word = wordcap
                        rc += 1

                    elif wordlow in entries_l:
                        word = wordlow
                        rc += 1


            if word in listexcept:
                word = "\""
                rc += 1

            if synonyms is not None:
                if word in synonyms.keys():
                    word = synonyms[word]
                    rc += 1

        linesx.append(word + opos + olema)

    print('lexicon magic word replacements: ' + str(rc))
    return linesx, origlines