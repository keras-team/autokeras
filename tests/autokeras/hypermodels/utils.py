import kerastuner

from autokeras import hypermodels


def name_in_hps(hp_name, hp):
    return any([hp_name in name for name in hp.values])


def block_basic_exam(block, inputs, hp_names):
    hp = kerastuner.HyperParameters()
    block = hypermodels.deserialize(hypermodels.serialize(block))
    outputs = block.build(hp, inputs)

    for hp_name in hp_names:
        assert name_in_hps(hp_name, hp)

    return outputs
