import kerastuner

from autokeras import graph as graph_module


def name_in_hps(hp_name, hp):
    return any([hp_name in name for name in hp.values])


def block_basic_exam(block, inputs, hp_names):
    hp = kerastuner.HyperParameters()
    block = graph_module.deserialize(graph_module.serialize(block))
    outputs = block.build(hp, inputs)

    for hp_name in hp_names:
        assert name_in_hps(hp_name, hp)

    return outputs
