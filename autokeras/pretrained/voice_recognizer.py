from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from autokeras.pretrained.base import Pretrained
from autokeras.constant import Constant

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        # e.g. labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, probs):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, return_offsets=True):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = []
        for sequence in sequences:
            seq_len = len(sequence)
            string, string_offsets = self.process_string(sequence, seq_len)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        return strings, offsets

    def process_string(self, sequence, size):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char == self.int_to_char[self.blank_index]:
                continue
                # if this char is a repetition and remove_repetitions=true, then skip
            if i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                continue
            if char == self.labels[self.space_index]:
                string += ' '
                offsets.append(i)
            else:
                string = string + char
                offsets.append(i)
        return string, torch.IntTensor(offsets)

    def decode(self, probs):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of seq_length x batch x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   return_offsets=True)
        return strings, offsets


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x


class InferenceBatchSoftmax(nn.Module):
    def __init__(self):
        super(InferenceBatchSoftmax, self).__init__()

    @staticmethod
    def forward(input_):
        return F.softmax(input_, dim=-1)


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5,
                 bidirectional=True):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._labels = labels
        self._bidirectional = bidirectional

        num_classes = len(self._labels)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(0, 10)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        # rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        # rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        # rnn_input_size *= 32
        rnn_input_size = 672

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package.get('bidirectional', True))
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model


class VoiceRecognizer(Pretrained):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = self.load_checkpoint()
        labels = Constant.VOICE_RECONGINIZER_LABELS
        self.decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    @property
    def _google_drive_files(self):
        return Constant.VOICE_RECONGINIZER_MODELS

    def load_checkpoint(self):
        model = DeepSpeech.load_model(self.local_paths[0], cuda=(self.device == 'cuda'))
        model.eval()
        return model

    def predict(self, audio_data, audio_path=None):
        if audio_data is None:
            raise TypeError("audio_data cannot be None")
        audio_data = audio_data.view(1, 1, audio_data.size(0), audio_data.size(1))
        with torch.no_grad():
            out = self.model(Variable(audio_data))
            out = out.transpose(0, 1)  # TxNxH
            decoded_output, _ = self.decoder.decode(out.data)
        return decoded_output[0][0]
