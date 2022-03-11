import os
from stanza import Pipeline
from stanza.models.pos.trainer import unpack_batch
from stanza.models.common.utils import unsort
from stanza.models.pos.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
import torch.nn.functional as F
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common import utils, loss
from stanza.utils.conll import CoNLL
from torch import nn, cat
from os import path

from beast.scripts.pipeline import get_sen_toks


def tag_stanza(par_path, file_path, out_path):

    tokens = get_sen_toks(file_path)
    words = [x for y in tokens for x in y]
    toknlp, nlp, labels = unroll_par(par_path)

    document = toknlp(tokens)
    scores, preds = getScores(nlp, document)

    with open(out_path, 'w', encoding='utf-8') as o:
        for c, word in enumerate(words):
            wl = word
            for i, x in enumerate(scores[c]):
                tag = labels.id2unit(i)
                score = round(x.item(), 4)
                if score > 0.01 and tag != '_SP':
                    wl += "\t" + tag + " " + str(score)
            o.write(wl+"\n")


def getScores(nlp, document):

    posproc = nlp.processors["pos"]
    sm = nn.Softmax(dim=1)
    trainer = posproc.trainer

    batch = DataLoader(document, 1, posproc.config, posproc.pretrain, vocab=posproc.vocab, evaluation=True,
                       sort_during_eval=True)
    scores = []
    thempreds = []

    for i, b in enumerate(batch):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(b, trainer.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs
        trainer.model.eval()
        batch_size = word.size(0)

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if trainer.model.args['word_emb_dim'] > 0:
            word_emb = trainer.model.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if trainer.model.args['pretrain']:
            pretrained_emb = trainer.model.pretrained_emb(pretrained)
            pretrained_emb = trainer.model.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if trainer.model.args['char'] and trainer.model.args['char_emb_dim'] > 0:
            char_reps = trainer.model.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(trainer.model.trans_char(trainer.model.drop(char_reps.data)),
                                       char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = cat([x.data for x in inputs], 1)
        lstm_inputs = trainer.model.worddrop(lstm_inputs, trainer.model.drop_replacement)
        lstm_inputs = trainer.model.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = trainer.model.taggerlstm(lstm_inputs, sentlens, hx=(
            trainer.model.taggerlstm_h_init.expand(2 * trainer.model.args['num_layers'], word.size(0),
                                                   trainer.model.args['hidden_dim']).contiguous(),
            trainer.model.taggerlstm_c_init.expand(2 * trainer.model.args['num_layers'], word.size(0),
                                                   trainer.model.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data

        upos_hid = F.relu(trainer.model.upos_hid(trainer.model.drop(lstm_outputs)))
        upos_pred = trainer.model.upos_clf(trainer.model.drop(upos_hid))

        scores.extend(pad(sm(upos_pred)))

        preds = [pad(upos_pred).max(2)[1]]

        upos = pack(upos).data
        loss = trainer.model.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if trainer.model.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(trainer.model.drop(hid))
        else:
            xpos_hid = F.relu(trainer.model.xpos_hid(trainer.model.drop(lstm_outputs)))
            ufeats_hid = F.relu(trainer.model.ufeats_hid(trainer.model.drop(lstm_outputs)))

            if trainer.model.training:
                upos_emb = trainer.model.upos_emb(upos)
            else:
                upos_emb = trainer.model.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(trainer.model.drop(hid), trainer.model.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(trainer.model.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(trainer.model.vocab['xpos'])):
                xpos_pred = clffunc(trainer.model.xpos_clf[i], xpos_hid)
                loss += trainer.model.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(trainer.model.xpos_clf, xpos_hid)
            padded_xpos_pred = pad(xpos_pred)
            max_value = padded_xpos_pred.max(2)[1]
            loss += trainer.model.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(max_value)

        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(trainer.model.vocab['feats'])):
            ufeats_pred = clffunc(trainer.model.ufeats_clf[i], ufeats_hid)
            loss += trainer.model.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(cat(ufeats_preds, 2))

        upos_seqs = [trainer.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]

        xpos_seqs = preds[1]
        feats_seqs = [trainer.vocab['feats'].unmap(sent) for sent in preds[2].tolist()]

        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in
                       range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)

        thempreds += pred_tokens

    thempreds = unsort(thempreds, batch.data_orig_idx)
    score_seqs = unsort(scores, batch.data_orig_idx)

    preds_flattened = [y[0] for x in thempreds for y in x]
    scores_flattened = [y for x in score_seqs for y in x]

    return scores_flattened, preds_flattened


def unroll_par(par_path):

    par = os.path.basename(par_path)
    pardir = os.path.dirname(par_path)
    pt = par_path + "/../standard.pt"
    parx = os.listdir(par_path)[0]

    toknlp = Pipeline(par, dir=pardir, processors='tokenize', tokenize_pretokenized=True, logging_level='FATAL')

    nlp = Pipeline(par, dir=pardir, processors='tokenize,pos', tokenize_pretokenized=True,
                          pos_model_path=par_path + "/" + parx, pos_pretrain_path=pt, logging_level='FATAL')

    labels = nlp.processors["pos"].trainer.vocab['upos']

    return toknlp, nlp, labels


def stanza_split(list, ratio):
    chunksize = len(list) * ratio
    list1 = []
    list2 = []
    devs = False
    for i, doc in enumerate(list):
        if i < chunksize:
            list1.append(doc)
        elif doc.split("\t")[0] == "1" or devs:
            devs = True
            list2.append(doc)
        else:
            list1.append(doc)
    return list1, list2


def stanza_conl(conl):
    conl2 = []
    for c in conl:
        d = c.split("\t")
        if len(d) != 10:
            conl2.append(c)
        else:
            try:
                d = c.split("\t")
                d[3], d[4] = d[4], d[3]
                conl2.append("\t".join(d))
            except:
                print(c)
    return conl2


def prepare_stanza(conlulines, tempfiles, out, traindir, devdir, parser, pt):
    traindir = out + traindir
    devdir = out + devdir

    if parser != "":
        contemp = out + "/conll_temp"
        with open(contemp, 'w', encoding='utf8') as f:
            f.write('\n'.join(conlulines))
        doc = CoNLL.conll2doc(contemp)
        doc = depparse(doc, parser, pt)
        conl = CoNLL.doc2conll_text(doc)
        conlulines = conl.split("\n")
        tempfiles.append(contemp)

    conlulines = stanza_conl(conlulines)

    train_stanza, dev_stanza = stanza_split(conlulines, 0.9)

    with open(traindir, 'w', encoding='utf8') as f:
        f.write('\n'.join(train_stanza))

    with open(devdir, 'w', encoding='utf8') as f:
        f.write('\n'.join(dev_stanza))

    tempfiles.append(traindir)
    tempfiles.append(devdir)


def depparse(doc, parser, pt):

    dir = path.dirname(__file__)
    nlp = Pipeline("stanzadp", dir=dir, processors='depparse', depparse_pretagged=True,
                   depparse_model_path=parser, depparse_pretrain_path=pt, logging_level='FATAL')

    doc = nlp(doc)
    return doc





