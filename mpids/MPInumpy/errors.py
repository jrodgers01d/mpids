class MPInumpyError(Exception):
        """ Base exception class for MPInumpy errors. """
        pass

class InvalidDistributionError(MPInumpyError):
        """ Exception class for when a unsupported distribution is encountered. """
        pass

class ValueError(MPInumpyError):
        """ Exception class for when a invalid value is encountered. """
        pass

class NotSupportedError(MPInumpyError):
        """ Exception class for when a numpy feature is not supported. """
        pass

class IndexError(MPInumpyError):
        """ Exception class for when an access index is invalid. """
        pass

class TypeError(MPInumpyError):
        """ Exception class for when invalid data type is supplied. """
        pass
