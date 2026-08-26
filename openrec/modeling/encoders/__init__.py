__all__ = ['build_encoder']


def build_encoder(config):
    from .relg import RELG

    support_dict = [
        'RELG',
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when encoder of rec model only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
