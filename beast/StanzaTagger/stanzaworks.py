import torch
from stanza.models.common import doc
from stanza.models.pos.trainer import unpack_batch
from stanza.models.common.utils import unsort
from stanza.models.pos.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
import torch.nn.functional as F
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common import utils, loss
from torch import nn


def getScores(nlp, document, probability, lemmatize):

    posproc = nlp.processors["pos"]
    sm = nn.Softmax(dim=1)
    trainer = posproc.trainer

    batch = DataLoader(
        #document, posproc.config['batch_size'], posproc.config, posproc.pretrain, vocab=posproc.vocab, evaluation=True,
        #sort_during_eval=True)
        document, 1, posproc.config, posproc.pretrain, vocab=posproc.vocab, evaluation=True,
        sort_during_eval=True)
    scores = ([])
    thempreds = []
    score_seqs = []

    for i, b in enumerate(batch):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, word_string = unpack_batch(b, trainer.use_cuda)
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

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
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
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(trainer.model.xpos_clf, xpos_hid)
            padded_xpos_pred = pad(xpos_pred)
            if trainer.inflectional_lexicon is not None:
                max_value = trainer.inflectional_lexicon.process(padded_xpos_pred, word_string)
            else:
                max_value = padded_xpos_pred.max(2)[1]
            loss += trainer.model.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(max_value)

        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(trainer.model.vocab['feats'])):
            ufeats_pred = clffunc(trainer.model.ufeats_clf[i], ufeats_hid)
            loss += trainer.model.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))

        upos_seqs = [trainer.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]

        if trainer.inflectional_lexicon is None:
            xpos_seqs = [trainer.vocab['xpos'].unmap(sent) for sent in preds[1].tolist()]
        else:
            xpos_seqs = preds[1]
        feats_seqs = [trainer.vocab['feats'].unmap(sent) for sent in preds[2].tolist()]

        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in
                       range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)

        thempreds += (pred_tokens)
    thempreds = unsort(thempreds, batch.data_orig_idx)
    #score_seqs = [ss for ss in scores[0].tolist()]
    score_seqs = unsort(scores, batch.data_orig_idx)

    if 'use_lexicon' in posproc.config:
        preds_flattened = []
        skip = iter(posproc.predetermined_punctuations(batch.doc.get([doc.XPOS])))
        for x in thempreds:
            for y in x:
                n = next(skip, None)
                assert n is not None
                if not n:
                    preds_flattened.append(y)
                else:
                    preds_flattened.append(['PUNCT', 'Z', '_'])
    else:
        preds_flattened = [y for x in thempreds for y in x]

    scores_flattened = [y for x in score_seqs for y in x]
    scores_flattened = [x for x in scores_flattened if sum(map(float, x)) > 0.5]

    batch.doc.set([doc.UPOS, doc.XPOS, doc.FEATS], preds_flattened)
    newdoc = batch.doc

    if lemmatize:
        lemproc = nlp.processors["lemma"]
        newdoc = lemproc.process(newdoc)

    return scores_flattened, preds_flattened, newdoc









