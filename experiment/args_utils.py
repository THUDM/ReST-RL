import argparse


def int_list(s: str):
    return [int(x) for x in s.split(",")]


def str_list(s: str):
    return [x for x in s.split(",")]


def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise ValueError("Value must be an integer or a float.")


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for value in values:
            key, val = value.split('=', 1)
            if val.isdigit():
                val = int(val)
            elif val.replace('.', '', 1).isdigit():
                val = float(val)
            elif val.lower() in ('true', 'false'):
                val = val.lower() == 'true'
            kwargs[key] = val
        setattr(namespace, self.dest, kwargs)
