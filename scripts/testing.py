from scripts.tagging import tag_complex
from scripts.pipeline import lexentries, training_prep, lexmagic
from scripts.conversion import convert as conv


def complex_test(tagger="", file="", lexiconmagic=True, transliterate=True, out_path="./data/output",
                 lex_path="./data/lexicon/default"):

    tempfiles = ([])
    words = ([])
    tags = {}

    lines, lemacol, tagsets, newline, colsn = training_prep(file)

    tagset_names = ([])

    for tagset in tagsets:
        tags[tagset] = ([])
        tagset_names.append(tagset)

    for line in lines:
        if line == '\n':
            words.append(line)
        else:
            cols = line.split("\t")
            words.append(cols[0])
            for i in range(1, colsn-1):
                tags[tagset_names[i-1]].append(cols[i])

    del lines

    # transliteration
    if transliterate:
        print('transliteration...')
        for i, line in enumerate(words):
            words[i] = conv(line, 'CYRtoLAT')

    # This procedure changes the training dataset based of a lexicon
    if lexiconmagic:
        # load all possible words from the lexicon (in uppercase, capitalized and lowercase)
        entries_u, entries_c, entries_l = lexentries(lex_path)
        # convert using lexicon magic
        words, orliglineslm = lexmagic(set(entries_u), set(entries_c), set(entries_l), words)

    for tagset in tagsets:
        modelname = tagset + ".pt"

        with open(out_path + '/temp_test', 'w', encoding='utf-8') as temp:
            temp.write('\n'.join(words))
            tempfiles.append(out_path + '/temp_test')

        tag_accus = {}

        # for a in accus:
        # tag_accus[a.split('\t')[0]] = float(a.split('\t')[1].rstrip('\n'))

        newtags, tagger_tags, probs, csv = tag_complex(tagger, "", [out_path + '/temp_test'], out_path, False, False,
                                               False, False, False, False, True, [modelname], {}, False)

        test_results(csv, tag_accus, newtags[modelname], tags[tagset], tagger_tags, probs[modelname], words)


def test_results(csv, tag_accu, tags, results, tagger_answers, probs, words=None):
    high_pos_t = 0
    high_pos_f = 0
    high_npos_t = 0
    high_npos_f = 0

    jury_pos_t = 0
    jury_pos_f = 0
    jury_npos_t = 0
    jury_npos_f = 0

    xjury_pos_t = 0
    xjury_pos_f = 0
    xjury_npos_t = 0
    xjury_npos_f = 0

    cplx_pos_t = 0
    cplx_pos_f = 0
    cplx_npos_t = 0
    cplx_npos_f = 0

    high_ans = ([])
    jury_ans = ([])
    xjury_ans = ([])
    t_probs = ([])
    f_probs = ([])

    high = True
    jury = True
    xjury = False

    if words is not None:
        words = list(word.rstrip('\n') for word in words if word not in ['\n', '', '\0'])

    with open(csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # All lines including the blank ones

        # print(str(len(lines)))
        # print(str(len(tags)))
        # print(str(len(results)))

        tagset = {}
        normdict = {}
        normdicf = {}
        taggersall = ([])

        for i, tag in enumerate(lines[0].rstrip('\n').split('\t')):
            if tag != 'result':
                tagset[i] = tag.split('__')[1]
                normdict[i] = 0
                normdicf[i] = 0
                tgr = tag.split('__')[0]
                taggersall.append(tgr)

        taggerset = list(set(taggersall))

        del lines[0]

        for idx, line in enumerate(lines):
            values = line.rstrip('\n').split('\t')
            realtag = results[idx].rstrip('\n')
            newtag = tags[idx].rstrip('\n')
            # values[0] = 0.000

            if newtag.split(':')[0] == realtag.split(':')[0]:
                t_probs.append(str(probs[idx]))
                cplx_pos_t += 1
            else:
                cplx_pos_f += 1
                f_probs.append(str(idx + 1) + "|" + str(newtag) + "|" + str(probs[idx]))
            if newtag == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                cplx_npos_t += 1

            else:
                cplx_npos_f += 1

            # if realtag != 'SENT' and realtag != 'PUNCT':

            for i, val in enumerate(values):
                values[i] = float(val)

            if high:
                index = values.index(max(values))
                high_guess = tagset[index]
                high_ans.append(high_guess)

                if high_guess.split(':')[0] == realtag.split(':')[0]:
                    high_pos_t += 1
                else:
                    high_pos_f += 1
                if high_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    high_npos_t += 1
                else:
                    high_npos_f += 1

            if jury:
                for i, v in enumerate(values):
                    thistag = tagset[i]
                    for j in range(1, len(tagset)):
                        if tagset[j] == thistag:
                            values[i] += values[j]

                jury_guess = tagset[values.index(max(values))]
                jury_ans.append(jury_guess)

                if jury_guess.split(':')[0] == realtag.split(':')[0]:
                    jury_pos_t += 1
                else:
                    jury_pos_f += 1
                if jury_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    jury_npos_t += 1
                else:
                    jury_npos_f += 1

            if xjury:

                for i, v in enumerate(values):
                    thistag = tagset[i]
                    for j in range(1, len(tagset)):
                        if tagset[j] == thistag:
                            if taggersall[i] + "_" + thistag in tag_accu.keys():
                                accu = tag_accu[taggersall[i] + "_" + thistag]
                            else:
                                accu = 0.5
                            values[i] += values[j] * accu
                xjury_guess = tagset[values.index(max(values))]
                xjury_ans.append(xjury_guess)

                if xjury_guess.split(':')[0] == realtag.split(':')[0]:
                    xjury_pos_t += 1
                else:
                    xjury_pos_f += 1
                if xjury_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    xjury_npos_t += 1
                else:
                    xjury_npos_f += 1

    # print("\t".join(t_probs))
    # print("\t".join(f_probs))
    tagger_rates = {}
    for ta in tagger_answers:

        tagger_rates["pos_t"] = 0
        tagger_rates["pos_f"] = 0
        tagger_rates["npos_t"] = 0
        tagger_rates["npos_f"] = 0

        for i, t in enumerate(tagger_answers[ta]):
            r = results[i].rstrip('\n')
            if t.split(':')[0] == r.split(':')[0]:
                tagger_rates["pos_t"] += 1
            else:
                tagger_rates["pos_f"] += 1
            if t == r or (t =="PUNCT" and r == "SENT") or (r == "PUNCT" and t == "SENT"):
                tagger_rates["npos_t"] += 1
            else:
                tagger_rates["npos_f"] += 1

        rate_pos = 100 / (tagger_rates["pos_t"] + tagger_rates["pos_f"]) * tagger_rates["pos_t"]
        rate_npos = 100 / (tagger_rates["npos_t"] + tagger_rates["npos_f"]) * tagger_rates["npos_t"]
        print(ta + '\t' + str(rate_pos) + '\t' + str(rate_npos))

    if high:
        high_rate_pos = 100 / (high_pos_t + high_pos_f) * high_pos_t
        high_rate_npos = 100 / (high_npos_t + high_npos_f) * high_npos_t
        print('high\t' + str(high_rate_pos) + '\t' + str(high_rate_npos))

    if jury:
        jury_rate_pos = 100 / (jury_pos_t + jury_pos_f) * jury_pos_t
        jury_rate_npos = 100 / (jury_npos_t + jury_npos_f) * jury_npos_t
        print('jury\t' + str(jury_rate_pos) + '\t' + str(jury_rate_npos))

    if xjury:
        xjury_rate_pos = 100 / (xjury_pos_t + xjury_pos_f) * xjury_pos_t
        xjury_rate_npos = 100 / (xjury_npos_t + xjury_npos_f) * xjury_npos_t
        print('xjury\t' + str(xjury_rate_pos) + '\t' + str(xjury_rate_npos))

    cplx_rate_pos = 100 / (cplx_pos_t + cplx_pos_f) * cplx_pos_t
    cplx_rate_npos = 100 / (cplx_npos_t + cplx_npos_f) * cplx_npos_t
    print('cplx\t' + str(cplx_rate_pos) + '\t' + str(cplx_rate_npos))

    with open(csv[:len(csv) - 4] + '_poredjenje.csv', 'w', encoding='utf-8') as f:
        for t in tagger_answers.keys():
            f.write(t + "\t")
        if high:
            f.write("high\t")
        if jury:
            f.write("jury\t")
        if xjury:
            f.write("xjury\t")
        f.write("complex\t")
        f.write("targetPOS")
        if words is not None:
            f.write("\ttargetWord")
        f.write("\n")

        for i, res in enumerate(results):
            for ta in tagger_answers:
                f.write(tagger_answers[ta][i] + "\t")
            if high:
                f.write(high_ans[i] + "\t")
            if jury:
                f.write(jury_ans[i] + "\t")
            if xjury:
                f.write(xjury_ans[i] + "\t")
            f.write(tags[i] + "\t")
            f.write(results[i])
            if words is not None:
                f.write("\t"+words[i])
            f.write('\n')

