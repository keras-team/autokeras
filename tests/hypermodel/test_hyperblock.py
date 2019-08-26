import autokeras as ak
import kerastuner

from autokeras.hypermodel import node


def test_lightgbm_classifier_block():
    input_node = ak.Input()
    hp = kerastuner.HyperParameters()
    lgbm_classifier = ak.LightGBMClassifierBlock()
    output_node = lgbm_classifier.build(hp=hp, inputs=input_node)
    assert isinstance(output_node[0], node.Node)
