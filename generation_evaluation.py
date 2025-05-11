from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
import argparse



sample_op = lambda x: sample_from_discretized_mix_logistic(x, 10)  # Match training nr_logistic_mix=10
def my_sample(model, gen_data_dir, sample_batch_size=25, obs=(3, 32, 32), sample_op=sample_op):
    model.eval()
    with torch.no_grad():
        for label in my_bidict:  # Assuming my_bidict has 4 classes
            print(f"Generating samples for: {label}...")
            # Generate 25 images per label
            labels=torch.tensor(my_bidict[label], device= device)
            sample_t = sample(model, sample_batch_size, obs, sample_op, conditional= True, labels=labels)  
            sample_t = rescaling_inv(sample_t)  # Convert back to [0, 1]
            # Save images to gen_data_dir with label prefix
            save_images(sample_t, os.path.join(gen_data_dir), label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_data_dir', type=str,
                        default="data/test", help='Location for the dataset')
    
    args = parser.parse_args()
    
    ref_data_dir = args.ref_data_dir
    gen_data_dir = os.path.join(os.path.dirname(__file__), "samples")
    BATCH_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

  
    # Load the trained model and generate images
    model = PixelCNN(nr_resnet=5, nr_filters=120, input_channels=3, conditional=True, nr_logistic_mix=10, condition_strength='mild')  # Match training params
    model = model.to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'model_mild/conditional_pixelcnn_load_model_24.pth')
    if os.path.exists(model_path):
        # Handle potential multi-GPU trained model
        state_dict = torch.load(model_path, map_location=device)
        # Remove 'module.' prefix if present (from DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print('Model parameters loaded from', model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    # Generate images 
    my_sample(model=model, gen_data_dir=gen_data_dir, sample_batch_size=8, obs=(3, 32, 32), sample_op=sample_op)
  
    
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))