
class Debug:
    def __init__(self, verbose, message):
        if verbose:
            print("[DEBUG]  {}".format(message))