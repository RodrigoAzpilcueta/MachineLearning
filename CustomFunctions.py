def get_data(path = '',rescale=False,preprocess = None):
    '''
   Load and preprocess a numpy array of data.
   =========
   Parameters:
   ---------
   path(str): 
      Path to the numpy array.
   rescale(boolean):
      If true divides the elements of the array
      by 255.
   preprocess(function):
      Aplies a custom transformation to the 
      numpy array.
   '''
    try:
        data = np.load(path)        
    except FileNotFoundError:
        print('Wrong Path')
        return
    if rescale:
      data = data/255
    else:
      pass
    if preprocess is not None:
      data = preprocess(data)
    else:
      pass 
    return data
