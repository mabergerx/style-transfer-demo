from style_transfer.nn_definition import *


def load_model(modelpath='style_transfer/21styles.model'):
    style_model = Net(ngf=128)
    style_model.load_state_dict(torch.load(modelpath))
    return style_model