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
