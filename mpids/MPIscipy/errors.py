class MPIscipyError(Exception):
    """ Base exception class for MPIscipy errors. """
    pass

class ValueError(MPIscipyError):
    """ Exception class for when a invalid value is encountered. """
    pass

class TypeError(MPIscipyError):
    """ Exception class for when invalid data type is supplied. """
    pass
