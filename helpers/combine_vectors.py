import numpy as np
def combine_vec(file1 = '/home/jsk/skill-prediction/data/COLING/Fine-FTCondensedData/trn_point_embs.npy',file2 = '/home/jsk/skill-prediction/data/COLING/New-FT-true-label-concatCondensedData/trn_point_embs.npy', take_mean= False):
    """
    Combine two vector files, either  by taking mean or by stacking one vector with the other.
    """

    arr1 = np.load(file1)
    arr2 = np.load(file2)
    print(arr1.shape)
    if take_mean == True:
        arr3 = (arr1+arr2)/2
    else:
        arr3 = np.concatenate((arr1,arr2),axis =0)
    print(arr3.shape)
    np.save('/home/jsk/skill-prediction/data/CAB-COLING/New-FT-meanCondensedData/trn_point_embs.npy',arr3)
    return arr3

combine_vec(take_mean = True)