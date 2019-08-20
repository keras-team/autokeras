import tensorflow as tf
import kerastuner
from kerastuner.applications import resnet
from kerastuner.applications import xception


def set_hp_value(hp, name, value):
    full_name = hp._get_full_name(name)
    hp.values[full_name] = value or hp.values[full_name]

# hm = resnet.HyperResNet(include_top=False, input_shape=(33, 33, 3))
hm = xception.HyperXception(include_top=False, input_shape=(33, 33, 3))
hp = kerastuner.HyperParameters()
# hp.Choice('version', ['v1', 'v2', 'next'], default='v2')
# hp.Choice('v1/conv4_depth', [6, 23], default=23)
# hp.Choice('next/conv3_depth', [4, 8], default=4)
# hp.Choice('next/conv4_depth', [6, 23, 36], default=23)
# set_hp_value(hp, 'version', 'next')
# set_hp_value(hp, 'next/conv3_depth', 8)
# set_hp_value(hp, 'next/conv4_depth', 23)
model = hm.build(hp)
model.summary()
print(hp.get_config()['values'])
