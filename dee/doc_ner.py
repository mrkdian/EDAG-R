import torch
from torch import nn
import torch.nn.functional as F
import math

from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel
from .ner_model import NERTokenEmbedding, CRFLayer
from . import transformer

class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out

class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)

class DocNERModel(nn.Module):
    def __init__(self, config):
        super(DocNERModel, self).__init__()

        self.config = config
        # Word Embedding, Word Local Position Embedding
        self.token_embedding = NERTokenEmbedding(
            config.vocab_size, config.hidden_size,
            max_sent_len=config.max_sent_len, dropout=config.dropout
        )
        # Multi-layer Transformer Layers to Incorporate Contextual Information
        self.token_encoder_1 = transformer.make_transformer_encoder(
            config.num_tf_layers // 2, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        )

        self.token_encoder_2 = transformer.make_transformer_encoder(
            config.num_tf_layers // 2, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
        )

        if self.config.use_crf_layer:
            self.crf_layer = CRFLayer(config.hidden_size, self.config.num_entity_labels)
        else:
            # Token Label Classification
            self.classifier = nn.Linear(config.hidden_size, self.config.num_entity_labels)

        # if self.config.seq_reduce_type == 'AWA':
        #     self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)

        # self.sent_pos_encoder = SentencePosEncoder(
        #     config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        # )

        self.cut_word_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, 2)
        )

    def forward(self, input_ids, input_masks, cut_word_label, doc_start_list, doc_batch_size,
                label_ids=None, train_flag=True, decode_flag=True):
        """Assume input size [batch_size, seq_len]"""
        if input_masks.dtype != torch.uint8:
            input_masks = input_masks == 1
        if train_flag:
            assert label_ids is not None

        # get contextual info
        # input_emb = self.token_embedding(input_ids)
        # input_masks = input_masks.unsqueeze(-2)  # to fit for the transformer code
        # batch_seq_enc = self.token_encoder_1(input_emb, input_masks)
        
        cut_word_loss = []
        cut_word_pred = []
        batch_seq_enc = []
        for batch_idx in range(doc_batch_size):
            idx_start = doc_start_list[batch_idx]
            idx_end = doc_start_list[batch_idx + 1]

            input_emb_1 = self.token_embedding(input_ids[idx_start: idx_end])
            input_mask_1 = input_masks[idx_start: idx_end].unsqueeze(-2)
            seq_enc_1 = self.token_encoder_1(input_emb_1, input_mask_1)

            # sent_ctx_1 = self.get_doc_sent_emb(seq_enc_1, input_mask_1).unsqueeze(0).expand(seq_enc_1.shape[0], -1, -1)
            # input_mask_2 = torch.ones(input_emb_2.shape[0], 1, input_emb_2.shape[0], dtype=torch.uint8, device=input_emb_2.device)
            # input_mask_2 = torch.cat([input_mask_2, input_mask_1], dim=-1)
            # input_emb_2 = torch.cat([sent_ctx_1, seq_enc_1], dim=1)
            input_emb_2 = seq_enc_1
            doc_seq_enc = self.token_encoder_2(input_emb_2, input_mask_1)
            doc_seq_enc = doc_seq_enc[:, :input_emb_1.shape[1]]
            batch_seq_enc.append(doc_seq_enc)

            if cut_word_label is not None:
                _cut_word_loss, _cut_word_pred = self.get_cut_word_loss(doc_seq_enc, input_mask_1, cut_word_label[idx_start: idx_end])
                cut_word_loss.append(_cut_word_loss)
                cut_word_pred.append(_cut_word_pred)
            else:
                cut_word_pred = None
                cut_word_loss = None
        batch_seq_enc = torch.cat(batch_seq_enc, dim=0)
        if cut_word_loss is not None:
            cut_word_loss  = torch.cat(cut_word_loss, dim=0)
            cut_word_pred = torch.cat(cut_word_pred, dim=0)


        if self.config.use_crf_layer:
            ner_loss, batch_seq_preds = self.crf_layer(
                batch_seq_enc, seq_token_label=label_ids, batch_first=True,
                train_flag=train_flag, decode_flag=decode_flag
            )
        else:
            # [batch_size, seq_len, num_entity_labels]
            batch_seq_logits = self.classifier(batch_seq_enc)
            batch_seq_logp = F.log_softmax(batch_seq_logits, dim=-1)

            if train_flag:
                batch_logp = batch_seq_logp.view(-1, batch_seq_logp.size(-1))
                batch_label = label_ids.view(-1)
                # ner_loss = F.nll_loss(batch_logp, batch_label, reduction='sum')
                ner_loss = F.nll_loss(batch_logp, batch_label, reduction='none')
                ner_loss = ner_loss.view(label_ids.size()).sum(dim=-1)  # [batch_size]
            else:
                ner_loss = None

            if decode_flag:
                batch_seq_preds = batch_seq_logp.argmax(dim=-1)
            else:
                batch_seq_preds = None
        
        if cut_word_label is not None and ner_loss is not None:
            cut_word_loss = cut_word_loss.sum(dim=-1)
            ner_loss += cut_word_loss
        return batch_seq_enc, ner_loss, batch_seq_preds, cut_word_pred

    def get_cut_word_loss(self, sent_emb, mask, target):
        pred = self.cut_word_classifier(sent_emb)
        pred = pred.view(pred.shape[0] * pred.shape[1], -1)
        # mask = mask.squeeze().view(-1)
        target = target.view(-1)
        loss = F.cross_entropy(pred, target, ignore_index=-100, reduction='none')
        return loss.view(sent_emb.shape[0], sent_emb.shape[1]), pred.argmax(dim=-1).view(sent_emb.shape[0], -1)

    def get_doc_sent_emb(self, ner_token_emb, ner_token_masks):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_emb = self.sent_pos_encoder(total_sent_emb)

        return total_sent_emb