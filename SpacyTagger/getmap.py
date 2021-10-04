def gettagmap(uniqpos):
    tagdic = {}
    for p in uniqpos:
        if 'PUN' in p or 'SENT' in p:
            tagdic[p] = 'PUNCT'
        elif 'ADJ' in p or p == 'A' or 'A:' in p:
            tagdic[p] = 'ADJ'
        elif 'ADP' in p or 'PRE' in p:
            tagdic[p] = 'ADP'
        elif 'ADV' in p:
            tagdic[p] = 'ADV'
        elif 'AUX' in p:
            tagdic[p] = 'AUX'
        elif 'CON' in p:
            tagdic[p] = 'CONJ'
        elif 'DET' in p:
            tagdic[p] = 'DET'
        elif 'INT' in p:
            tagdic[p] = 'INTJ'
        elif 'NOUN' in p or p == 'N' or 'N:' in p:
            tagdic[p] = 'NOUN'
        elif 'NUM' in p:
            tagdic[p] = 'NUM'
        elif 'PAR' in p:
            tagdic[p] = 'PART'
        elif 'PROP' in p:
            tagdic[p] = 'PROPN'
        elif 'PRO' in p:
            tagdic[p] = 'PRON'
        elif 'SCON' in p:
            tagdic[p] = 'SCONJ'
        elif 'SYM' in p:
            tagdic[p] = 'SYM'
        elif 'VERB' in p or p == 'V' or 'V:' in p:
            tagdic[p] = 'VERB'
        elif 'SPA' in p:
            tagdic[p] = 'SPACE'
        else:
            tagdic[p] = 'X'
    return tagdic
