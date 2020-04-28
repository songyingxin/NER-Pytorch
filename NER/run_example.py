import os


from config import config
from NER.utils.utils import get_device, set_seed
# from NER.utils import get_device, set_seed

def main(args):

    # overwrite output_dir or not 
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train):
        print("Output directory ({}) already exists and is not empty. ".format(
            args.output_dir))
        print("Do you want overwrite it? type y or n")
        
        if input() == 'n':
            return

    # device ready
    gpu_ids = [int(device_id) for device_id in args.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    args.n_gpu = n_gpu

    # set random seed
    set_seed(args)


if __name__ == "__main__":

    main(config())
    
