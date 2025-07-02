import math

import torch
import torch.nn.functional as F
from torch import nn

from attention import Attention
from embedding import MyEmbedding, PositionalEncoding
from encoder import TransEncoder
from fullyconnect import FullyConnect


class MyModel(nn.Module):
    def __init__(self, config, args):
        super(MyModel, self).__init__()
        self.config = config
        self.args = args
        self.base_dim = args.base_dim
        self.prefer_dim = args.prefer_dim
        self.num_locations = config.Dataset.num_locations
        self.num_timeslot = args.num_timeslots
        self.embedding_layer = MyEmbedding(config, args)
        self.num_prototypes = args.num_prototypes

        emb_dim = self.base_dim*1
        self.pe = PositionalEncoding(emb_dim=emb_dim)
        self.encoder = TransEncoder(emb_dim)

        self.encoder_time = TransEncoder(emb_dim)

        if self.num_prototypes > 0:
            self.attention_loc = Attention(args)


        time_head_input_dim = self.base_dim *2
        self.time_head = FullyConnect(input_dim=time_head_input_dim,
                                        output_dim=self.num_timeslot)
        self.convert_time = nn.Sequential(
            nn.Linear(self.num_timeslot, self.base_dim),
        )

        location_head_input_dim = self.base_dim * 2 + self.prefer_dim
        if self.tpl:
            location_head_input_dim -= self.base_dim

        self.location_head = FullyConnect(input_dim=location_head_input_dim,
                                          output_dim=self.num_locations)
        self.drop = nn.Dropout(0.1)



    def forward(self, batch_data, aux_mat):
        loc_ids = batch_data['location_x']

        batch_size, sequence_length = loc_ids.shape
        device = self.embedding_layer.user_embedding.weight.device

        loc_emb_all, time_emb, user_emb, loc_proto_emb_enhanced, prefer_emb, _ = self.embedding_layer(batch_data, aux_mat)

        loc_emb = loc_emb_all[loc_ids]

        lt_emb = loc_emb + time_emb

        future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(device)
        future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
        encoder_out = self.encoder(self.pe(lt_emb * math.sqrt(lt_emb.size(-1))), src_mask=future_mask)

        prefer_emb = prefer_emb.unsqueeze(1).repeat(1, sequence_length, 1)
        if self.num_prototypes > 0:
            loc_head_input = torch.cat([encoder_out, prefer_emb], dim=-1)
        else:
            loc_head_input = torch.cat([encoder_out, user_emb], dim=-1)

        ut_emb = time_emb + user_emb

        if self.num_prototypes > 0:
            proto_q = loc_head_input
            _, semantic_logits = self.attention_loc(q=proto_q, k=loc_proto_emb_enhanced)

        encoder_out_time = self.encoder_time(self.pe(ut_emb * math.sqrt(ut_emb.size(-1))), src_mask=future_mask)
        time_head_input = torch.cat([encoder_out_time, loc_emb], dim=-1)
        time_logits = self.time_head(time_head_input.view(batch_size * sequence_length, -1))
        time_probs = F.softmax(time_logits, dim=-1)

        next_time_mixed_emb = self.convert_time(time_logits)
        loc_head_input = loc_head_input.view(batch_size * sequence_length, -1)
        loc_head_input = torch.cat([loc_head_input, next_time_mixed_emb], dim=-1)

        loc_logits = self.location_head(loc_head_input)
        loc_logits = loc_logits.view(batch_size, sequence_length, -1)

        if self.num_prototypes > 0:

            adjust_loc_logits = loc_logits + semantic_logits
            adjust_loc_probs = F.softmax(adjust_loc_logits, dim=-1)
        else:
            adjust_loc_probs = F.softmax(loc_logits, dim=-1)

        adjust_loc_probs = adjust_loc_probs.view(batch_size * sequence_length, -1)

        return torch.log(adjust_loc_probs), torch.log(time_probs)
