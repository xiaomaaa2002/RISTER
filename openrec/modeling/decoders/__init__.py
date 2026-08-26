__all__ = ['build_decoder']


def build_decoder(config):
    # rec decoder
    from .ritext_decoder import RITextDecoder

    support_dict = [
        'RITextDecoder',
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'decoder only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
