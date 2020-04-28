import torch
import random
import numpy as np

def get_device(gpu_id):
    """ get the devive: cpu or gpu
    
    Returns:
        device: gpu or cpu device
        n_gpu: the gpu number in the machine
    """
    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, the avaiable gpu num is {}".format(n_gpu))
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def set_seed(args):
    """set random seed
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)