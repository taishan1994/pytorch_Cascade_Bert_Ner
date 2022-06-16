import torch
import torch.nn as nn
from bert_base_model import BaseModel
from torchcrf import CRF
import config


class BertNerModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        super(BertNerModel, self).__init__(bert_dir=args.bert_dir, dropout_prob=args.dropout_prob)
        self.args = args
        self.num_layers = args.num_layers
        self.lstm_hidden = args.lstm_hidden
        gpu_ids = args.gpu_ids.split(',')
        device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.device = device

        out_dims = self.bert_config.hidden_size

        if args.use_lstm == 'True':
            self.lstm = nn.LSTM(out_dims, args.lstm_hidden, args.num_layers, bidirectional=True,batch_first=True, dropout=args.dropout)
            self.bio_linear = nn.Linear(args.lstm_hidden * 2, args.bio_tags)
            self.att_linear = nn.Linear(args.lstm_hidden * 2, args.att_tags)
            self.criterion = nn.CrossEntropyLoss()
            init_blocks = [self.bio_linear, self.att_linear]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        else:
            mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, mid_linear_dims),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            #
            out_dims = mid_linear_dims

            # self.dropout = nn.Dropout(dropout_prob)
            # self.classifier = nn.Linear(out_dims, args.num_tags)
            self.bio_linear = nn.Linear(args.lstm_hidden * 2, args.bio_tags)
            self.att_linear = nn.Linear(args.lstm_hidden * 2, args.att_tags)
            # self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.criterion = nn.CrossEntropyLoss()


            init_blocks = [self.mid_linear, self.bio_linear, self.att_linear]
            # init_blocks = [self.classifier]
            self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

        if args.use_crf == 'True':
            self.crf = CRF(args.bio_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
        return h0, c0

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                bio_labels,
                att_labels):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)

        if self.args.use_lstm == 'True':
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
            bio_seq_out = self.bio_linear(seq_out)
            att_seq_out = self.att_linear(seq_out)
            bio_seq_out = bio_seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1) #[batchsize, max_len, num_tags]
            att_seq_out = att_seq_out.contiguous().view(batch_size, self.args.max_seq_len, -1) #[batchsize, max_len, num_tags]
        else:
            seq_out = self.mid_linear(seq_out)  # [batchsize, max_len, 256]
            # seq_out = self.dropout(seq_out)
            # seq_out = self.classifier(seq_out)  # [24, 256, 53]
            bio_seq_out = self.bio_linear(seq_out)
            att_seq_out = self.att_linear(seq_out)

        if self.args.use_crf == 'True':
            # bio_logits = self.crf.decode(bio_seq_out, mask=attention_masks)
            if bio_labels is None:
                # att_logits = torch.argmax(att_seq_out, -1)
                return bio_seq_out, att_seq_out

            active = torch.argmax(bio_seq_out, -1).view(-1) > 0  # 这里取出为实体的部分
            active_att_logits = att_seq_out.view(-1, att_seq_out.size()[2])[active]
            active_att_labels = att_labels.view(-1)[active]
            
            bio_loss = -self.crf(bio_seq_out, bio_labels, mask=attention_masks, reduction='mean')
            att_loss = self.criterion(active_att_logits, active_att_labels)
            loss = bio_loss + att_loss
            outputs = (loss, ) + (bio_seq_out, att_seq_out)
            return outputs
        else:
            if bio_labels is None:
                return bio_seq_out, att_seq_out
            bio_active = attention_masks.view(-1) == 1
            att_active = torch.argmax(bio_seq_out, -1).view(-1) > 0
            active_bio_logits = bio_seq_out.view(-1, bio_seq_out.size()[2])[bio_active]
            active_bio_labels = bio_labels.view(-1)[bio_active]
            active_att_logits = att_seq_out.view(-1, att_seq_out.size()[2])[att_active]
            active_att_labels = att_labels.view(-1)[att_active]
            bio_loss = self.criterion(active_bio_logits, active_bio_labels)
            att_loss = self.criterion(active_att_logits, active_att_labels)
            loss = bio_loss + att_loss
            outputs = (loss,) + (bio_seq_out, att_seq_out)
            return outputs

if __name__ == '__main__':
    args = config.Args().get_parser()
    args.bio_tags = 4
    args.att_tags = 10
    args.use_lstm = 'True'
    args.use_crf = 'True'

    model = BertNerModel(args)
    for name,weight in model.named_parameters():
        print(name)