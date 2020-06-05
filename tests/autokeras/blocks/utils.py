import kerastuner

from autokeras import blocks
from tests import utils


def name_in_hps(hp_name, hp):
    return any([hp_name in name for name in hp.values])


def block_basic_exam(block, inputs, hp_names):
    hp = kerastuner.HyperParameters()
    block = blocks.deserialize(blocks.serialize(block))
    utils.config_tests(block, excluded_keys=[
        'inputs',
        'outputs',
        'build',
        '_build',
        'input_tensor',
        'input_shape',
        'include_top',
        '_num_output_node'])
    outputs = block.build(hp, inputs)

    for hp_name in hp_names:
        assert name_in_hps(hp_name, hp)

    return outputs
