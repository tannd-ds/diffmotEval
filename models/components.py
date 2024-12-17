import torch
from torch import nn
from einops import rearrange, repeat
import math


class ResidualConnection(nn.Module):
    def __init__(self, _size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(_size)

    def forward(self, x, att_features):
        return x + self.dropout(self.norm(att_features))


class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.dense_layer = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        normalized_x = self.layer_norm(x)
        ffn_x = self.dense_layer(normalized_x)
        output_x = self.dropout(ffn_x)
        return output_x


class TransAoA(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size=256):
        super(TransAoA, self).__init__()
        layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                           nhead=1,
                                           dim_feedforward=4 * hidden_size)
        layer_norm = nn.LayerNorm(hidden_size)
        self.mlp_input = MLP(in_features=input_size, out_features=hidden_size)
        self.transformer_core = nn.TransformerDecoder(decoder_layer=layer,
                                                      num_layers=num_layers,
                                                      norm=layer_norm)
        self.aoa = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GLU(),
        )  # AoA Layer

        self.residual_fn = ResidualConnection(hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, input, ctx):
        # Check if input and ctx are 2D, if so, unsqueeze to make them 3D
        # if input.dim() == 2:
        #     input = input.unsqueeze(1)
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(1)

        input = self.mlp_input(input)

        encoded_input = self.transformer_core(tgt=input,
                                              memory=ctx)
        aoa_output = self.aoa(torch.cat([encoded_input, input], dim=-1))
        res_connection = self.residual_fn(input, aoa_output)

        output = self.head(res_connection)  # Squeeze the output back to 2D if the input was originally 2D
        # if output.size(1) == 1:
        #     output = output.squeeze(1)

        return output


class ReUnet3PlusDownBlock(nn.Module):
    def __init__(self, down_num, up_num, mid_num, filter, reversed_filters, num_layers, hidden_size=256):
        super(ReUnet3PlusDownBlock, self).__init__()

        # Down, Up, Mid Sampling
        self.down_num = down_num
        self.up_num = up_num
        self.mid_num = mid_num
        self.down_samplers, self.up_samplers, self.mid_samplers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        # Down Sampling
        for i in range(self.down_num):
            block = TransAoA(input_size=reversed_filters[i],
                             output_size=reversed_filters[self.down_num],
                             num_layers=num_layers, )
            self.down_samplers.append(block)

        # Up Sampling
        for i in range(self.up_num):
            block = TransAoA(input_size=filter[i],
                             output_size=reversed_filters[self.down_num],
                             num_layers=num_layers, )
            self.up_samplers.append(block)

        # Mid Sampling
        self.mid_samplers = TransAoA(input_size=reversed_filters[self.down_num],
                                     output_size=reversed_filters[self.down_num],
                                     num_layers=num_layers, )

        self.output = TransAoA(input_size=reversed_filters[self.down_num] * 4,
                               output_size=reversed_filters[self.down_num],
                               num_layers=num_layers, )

    def forward(self, down_samples, up_samples, mid_samples, ctx):
        # Down Sampling
        down_list = []
        for i, block in enumerate(self.down_samplers):
            down_output = block(input=down_samples[i],
                                ctx=ctx)
            down_list.append(down_output)

        # Up Sampling
        up_list = []
        for i, block in enumerate(self.up_samplers):
            up_output = block(input=up_samples[i],
                              ctx=ctx)
            up_list.append(up_output)

        # Mid Sampling
        mid_list = []
        mid_output = self.mid_samplers(input=mid_samples,
                                       ctx=ctx)
        mid_list.append(mid_output)

        # Output
        all_samples = up_list + mid_list + down_list
        concat_samples = torch.stack(all_samples, dim=1)
        concat_samples = concat_samples.view(-1, concat_samples.size(1) * concat_samples.size(2))
        return self.output(input=concat_samples,
                           ctx=ctx)


class ReUnet3PlusDownBlock_Smaller(nn.Module):
    def __init__(self, down_num, up_num, mid_num, filter, reversed_filters, num_layers, hidden_size=256):
        super(ReUnet3PlusDownBlock_Smaller, self).__init__()

        # Down, Up, Mid Sampling
        self.down_num = down_num
        self.up_num = up_num
        self.mid_num = mid_num
        self.down_samplers, self.up_samplers, self.mid_samplers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        # Down Sampling
        for i in range(self.down_num):
            block = MLP(in_features=reversed_filters[i],
                        out_features=reversed_filters[self.down_num], )
            self.down_samplers.append(block)

        # Up Sampling
        for i in range(self.up_num):
            block = MLP(in_features=filter[i],
                        out_features=reversed_filters[self.down_num], )
            self.up_samplers.append(block)

        self.output = TransAoA(input_size=reversed_filters[self.down_num] * 4,
                               output_size=reversed_filters[self.down_num],
                               num_layers=num_layers, )

    def forward(self, down_samples, up_samples, mid_samples, ctx):
        # Down Sampling
        down_list = []
        for i, block in enumerate(self.down_samplers):
            down_output = block(down_samples[i])
            down_list.append(down_output)

        # Up Sampling
        up_list = []
        for i, block in enumerate(self.up_samplers):
            up_output = block(up_samples[i])
            up_list.append(up_output)

        # Skip Connection
        mid_list = []
        mid_list.append(mid_samples)

        # Output
        all_samples = up_list + mid_list + down_list
        concat_samples = torch.stack(all_samples, dim=1)
        concat_samples = concat_samples.view(-1, concat_samples.size(1) * concat_samples.size(2))
        return self.output(input=concat_samples,
                           ctx=ctx)