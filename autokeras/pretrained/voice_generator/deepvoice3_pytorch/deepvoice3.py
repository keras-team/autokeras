# coding: utf-8

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modules import Conv1d, ConvTranspose1d, Embedding, Linear
from .modules import SinusoidalEncoding, Conv1dGLU


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,
                 max_positions=512, dropout=0.1, apply_grad_scaling=False):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling

        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, embed_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = None

        input_embedding = x

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        keys = x.transpose(1, 2)

        # scale gradients (this only affects backward, not forward)
        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, dropout=0.1,
                 window_ahead=3, window_backward=1,
                 key_projection=True, value_projection=True):
        super(AttentionLayer, self).__init__()
        self.query_projection = Linear(conv_channels, embed_dim)
        self.key_projection = None
        self.value_projection = None
        self.out_projection = Linear(embed_dim, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query

        # attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

        mask_value = -float("inf")

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_speakers, speaker_embed_dim,
                 in_dim=80, r=5,
                 max_positions=512, padding_idx=None,
                 preattention=((128, 5, 1),) * 4,
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=True,
                 value_projection=True,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        in_channels = in_dim * r

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, convolutions[0][0])
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings
        self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim * r
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=False))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               key_projection=key_projection,
                               value_projection=value_projection)
                if attention[i] else None)
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, in_dim * r, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None):
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions, speaker_embed)
            return outputs

        # Grouping multiple frames if necessary

    def incremental_forward(self, encoder_out, text_positions, speaker_embed=None,
                            initial_input=None, test_inputs=None):
        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        w = self.key_position_rate
        text_pos_embed = self.embed_keys_positions(text_positions, w)
        keys = keys + text_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim * self.r).zero_()
        current_input = initial_input
        while True:
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_pos, w)

            if t > 0:
                current_input = outputs[-1]
            x = current_input
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Prenet
            for f in self.preattention:
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                residual = x
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)

                if attention is not None:
                    assert isinstance(f, Conv1dGLU)
                    x = x + frame_pos_embed
                    x, alignment = attention(x, (keys, values),
                                             last_attended=last_attended[idx])
                    if self.force_monotonic_attention[idx]:
                        last_attended[idx] = alignment.max(-1)[1].view(-1).data[0]
                    if ave_alignment is None:
                        ave_alignment = alignment
                    else:
                        ave_alignment = ave_alignment + ave_alignment

                # residual
                if isinstance(f, Conv1dGLU):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            x = self.last_conv.incremental_forward(x)
            ave_alignment = ave_alignment.div_(num_attention_layers)

            # Ooutput & done flag predictions
            output = F.sigmoid(x)
            done = F.sigmoid(self.fc(x))

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]

            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments, dones, decoder_states

    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        self.last_conv.clear_buffer()


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, out_dim, convolutions=((256, 5, 1),) * 4,
                 time_upsampling=1,
                 dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = convolutions[0][0]
        # Idea from nyanko
        self.convolutions = nn.ModuleList([
            Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                   std_mul=1.0),
            ConvTranspose1d(in_channels, in_channels, kernel_size=2,
                            padding=0, stride=2, std_mul=1.0),
            Conv1dGLU(n_speakers, speaker_embed_dim,
                      in_channels, in_channels, kernel_size=3, causal=False,
                      dilation=1, dropout=dropout, std_mul=1.0, residual=True),
            Conv1dGLU(n_speakers, speaker_embed_dim,
                      in_channels, in_channels, kernel_size=3, causal=False,
                      dilation=3, dropout=dropout, std_mul=4.0, residual=True),
            ConvTranspose1d(in_channels, in_channels, kernel_size=2,
                            padding=0, stride=2, std_mul=4.0),
            Conv1dGLU(n_speakers, speaker_embed_dim,
                      in_channels, in_channels, kernel_size=3, causal=False,
                      dilation=1, dropout=dropout, std_mul=1.0, residual=True),
            Conv1dGLU(n_speakers, speaker_embed_dim,
                      in_channels, in_channels, kernel_size=3, causal=False,
                      dilation=3, dropout=dropout, std_mul=4.0, residual=True),
        ])

        std_mul = 4.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, out_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, x, speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        speaker_embed_btc = None
        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
        # Back to B x T x C
        x = x.transpose(1, 2)

        return F.sigmoid(x)
