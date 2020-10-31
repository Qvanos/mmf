
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.models.mmbt import MMBTBase, MMBTForPreTraining
from mmf.utils.modeling import get_optimizer_parameters_for_bert

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_bert import BertForPreTraining, BertPredictionHeadTransform

from mmf.common.registry import registry
from mmf.datasets.processors import VocabProcessor


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout_p=0.5, have_last_bn=False, pretrained_model_path=''):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], out_size)
        self.have_last_bn = have_last_bn

        if have_last_bn:
            self.bn = nn.BatchNorm1d(out_size)

        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)

        if self.have_last_bn:
            h = self.bn(h)

        return h


class SyntaxMultilevelEncoding(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(SyntaxMultilevelEncoding, self).__init__()

        self.config = config

        self.text_norm = config.text_norm
        self.word_dim = config.word_dim
        self.we_parameter = config.we_parameter
        self.concate = config.concate

        #embedding layer
        self.embed = nn.Embedding(config.vocab_size, config.word_dim)

        mapping_in_size = 0
        self.concat_bow = 'bow' in self.concate
        self.concat_pool = 'pool' in self.concate
        self.concat_rnn = 'gru' in self.concate
        self.concat_cnn = 'cnn' in self.concate
        if self.concat_bow:
          mapping_in_size += config.vocab_size

        if self.concat_pool:
          # 1-d convolutional network
          pool_channels = 512
          self.embed_conv = nn.Conv1d(in_channels=1,
                                      out_channels=config.pool_channels,
                                      kernel_size=3*config.word_dim,
                                      stride=config.word_dim)

          mapping_in_size += config.pool_channels

        if self.concat_rnn:
          # bidirectional rnn encoder
          self.rnn = nn.GRU(config.word_dim, config.text_rnn_size, batch_first=True, bidirectional=True)
          self.rnn_output_size = config.text_rnn_size*2

          mapping_in_size += self.rnn_output_size

        if self.concat_cnn:
          # visual 1-d convolutional network
          self.convs1 = nn.ModuleList([
              nn.Conv2d(1, config.cnn_out_channels, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
              for window_size in config.cnn_kernel_sizes
              ])

          mapping_in_size += config.cnn_out_channels * len(config.cnn_kernel_sizes)

        self.text_mapping = MLP(in_size=mapping_in_size,
                                hidden_sizes=config.mapping_hidden_sizes,
                                out_size=config.out_size,
                                dropout_p=config.dropout_p,
                                have_last_bn=config.have_last_bn)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, texts, lengths):
      # Level 1. Global Encoding by Max Pooling According
      embeddings = self.embed(texts)

      features = []

      if self.concat_bow:
        features.append(cap_bows)

      if self.concat_pool:
        embed_pool = embeddings.view(embeddings.shape[0], 1, -1)
        embed_pool = torch.relu(self.embed_conv(embed_pool))
        embed_pool = embed_pool.max(dim=2)[0]
        features.append(embed_pool)

      if self.concat_rnn:
        # Level 2. Temporal-Aware Encoding by biGRU
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        rnn_out, rnn_h = self.rnn(packed)
        rnn_out, lens_unpacked = pad_packed_sequence(rnn_out, batch_first=True)
        rnn_h = torch.cat([rnn_h[0,:, :], rnn_h[1,:,:]], dim=1)
        features.append(rnn_h)

      if self.concat_cnn:
        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = rnn_out.unsqueeze(1)
        con_out = [torch.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        features.append(con_out)

      # Levels' outputs concatenation
      features = torch.cat(features, dim=1)

      # mapping to common space
      features = self.text_mapping(features)
      if self.text_norm:
          features = l2norm(features)

      return features


class MMBTPOSForClassification(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config

        self.pos_encoder = SyntaxMultilevelEncoding(config.syntax_encoder, *args, **kwargs)

        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)

        self.bert_transform = BertPredictionHeadTransform(self.encoder_config)

        fc_in_size = self.encoder_config.hidden_size+config.syntax_encoder.out_size
        if config.use_polarity:
          fc_in_size + 4

        self.fc = nn.Linear(fc_in_size, self.config.num_labels)

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        pooled_output = module_output[1]
        output = {}

        if (
            self.encoder_config.output_hidden_states
            or self.encoder_config.output_attentions
        ):
            output["extras"] = module_output[2:]

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.bert_transform(pooled_output)

        syntax_output = self.pos_encoder(sample_list['pos_text'], sample_list['length'])

        concat_output = torch.cat((pooled_output, syntax_output), dim=1)

        if self.config.use_polarity:
          #polarity = sample_list['polarity'][:, -1].unsqueeze(1)
          polarity = sample_list['polarity']
          concat_output = torch.cat((polarity, concat_output), dim=1)

        logits = self.fc(concat_output)
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)
        output["scores"] = reshaped_logits

        return output


@registry.register_model("mmbt_pos")
class MMBTPOS(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = MMBTForPreTraining(self.config)
        else:
            self.model = MMBTPOSForClassification(self.config)

        if self.config.freeze_complete_base or self.config.freeze_text:
            for p in self.model.bert.mmbt.transformer.parameters():
                p.requires_grad = False

        if self.config.freeze_complete_base or self.config.freeze_modal:
            for p in self.model.bert.mmbt.modal_encoder.parameters():
                p.requires_grad = False

    # Backward compatibility for code from older mmbt
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("base.bert", "model.bert")
            .replace("base.cls", "model.cls")
            .replace("base.classifier", "model.classifier")
        )

    @classmethod
    def config_path(cls):
        return "configs/models/mmbt_pos/pretrain.yaml"

    def forward(self, sample_list):
        return self.model(sample_list)

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)
