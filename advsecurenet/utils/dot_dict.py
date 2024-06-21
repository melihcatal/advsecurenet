class DotDict(dict):
    """dot.notation access to dictionary attributes
    This class allows for dot notation access to dictionary attributes. 
    This additional functionality does not affect the existing dictionary methods.

    Taken from: https://stackoverflow.com/a/23689767/5768407
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
