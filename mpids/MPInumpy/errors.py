class MPInumpyError(Exception):
        """ Base exception class for MPInumpy errors. """
        pass

class InvalidDistributionError(MPInumpyError):
        """ Exception class for when a unsupported distribution is encountered """
        pass
