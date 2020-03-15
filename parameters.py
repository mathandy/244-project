import os
from os.path import join as fpath
from time import time


class AttributeDict(dict):
    """A class for dictionaries whose items also act like attributes

    Credit:
        https://stackoverflow.com/questions/4984647
    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


_run_type = 'unknown'
_run_time = str(int(round(time())))

_STATIC_DEFAULTS = AttributeDict({
    'epochs': 3,
    'batch_size': 1,
    'learning_rate': 0.0001,
    'run_name': _run_time,
    'input_length': None,
    'results_dir_root': os.path.expanduser('~/244-project-results'),
    'log_dir_root': os.path.expanduser('~/244-project-logs')
})


_DYNAMIC_DEFAULTS = AttributeDict({
    'results_dir': lambda args_: fpath(
        args_.results_dir_root, args_.run_type, args_.run_name),
    'log_dir': lambda args_: fpath(
        args_.log_dir_root, args_.run_type, args_.run_name),
    'shuffle_buffer_size': lambda args_: 4 * args_.batch_size,
})


def derive_dynamic_args(args, dynamic_defaults=_DYNAMIC_DEFAULTS):

    for parameter_name, derivation_fcn in dynamic_defaults.items():
        if args.get(parameter_name, None) is None:
            args[parameter_name] = derivation_fcn(args)
    return args


def get_run_parameters(custom_static_defaults={},
                       custom_dynamic_defaults={}):
    _STATIC_DEFAULTS.update(custom_static_defaults)
    _DYNAMIC_DEFAULTS.update(custom_dynamic_defaults)

    args = _STATIC_DEFAULTS
    return derive_dynamic_args(args)
