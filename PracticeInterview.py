import numpy as np

def oddtimes(arr):
    dict = {}
    for i in range(len(arr)):
        current = arr[i]
        
    result = np.bincount(arr)
    print(result)


oddtimes([1,2,2,3,3,3,9,9,9])