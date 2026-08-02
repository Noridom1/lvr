"""Compatibility shim for an optional module omitted from the public release."""


def make_packed_supervised_data_module_lvr_fixedToken(*args, **kwargs):
    raise NotImplementedError(
        "The fixed-token packed dataset implementation is not included in the public repository."
    )
