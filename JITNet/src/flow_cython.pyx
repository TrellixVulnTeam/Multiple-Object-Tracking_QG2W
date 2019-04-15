cimport cython
import numpy as np

DTYPE = np.int8

@cython.boundscheck(False)
@cython.wraparound(False)
def flow(char[:, :] pred, int[:, :, :] pred_flow):
    
    cdef Py_ssize_t height = pred.shape[0]
    cdef Py_ssize_t width = pred.shape[1]
    
    result = np.zeros((height, width), dtype=DTYPE)
    cdef char[:, :] result_view = result

    cdef int dx, dy
    cdef Py_ssize_t i, j
    
    for i in range(height):
        for j in range(width):
            dx = pred_flow[i, j, 0]
            dy = pred_flow[i, j, 1]
            if dx == 0 and dy == 0:
                result_view[i, j] = pred[i, j]
            else:
                x = j + dx
                y = i + dy
                if x >= 0 and y >= 0 and x < width and y < height:
                    result_view[y, x] = pred[i, j]
    return result