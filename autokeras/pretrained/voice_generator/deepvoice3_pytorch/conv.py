# coding: utf-8
from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None

    def incremental_forward(self, input_data):

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input_data.size(0)  # conv_input: bsz x len x dim
        if kw > 1:
            input_data = input_data.data
            if self.input_buffer is None:
                self.input_buffer = input_data.new(bsz, kw + (kw - 1) * (dilation - 1), input_data.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input_data[:, -1, :]
            input_data = self.input_buffer
            if dilation > 1:
                input_data = input_data[:, 0::dilation, :].contiguous()
        input_data = F.linear(input_data.view(bsz, -1), weight, self.bias)
        return input_data.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            weight = self.weight.transpose(1, 2).contiguous()

            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight
