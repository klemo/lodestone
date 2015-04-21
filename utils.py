import time

#------------------------------------------------------------------------------
    
def timeit(method):
    '''
    Simple profiling decorator
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Exec time for method --{}--: {:.2f} mins'.format(
                method.__name__, (te-ts)/60))
        return result
    return timed
