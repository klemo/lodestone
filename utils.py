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
        print('t={:.2f} mins'.format((te-ts)/60))
        return result
    return timed
