from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

def get_label(model, model_input, device):
   
    batch_size = model_input.shape[0]
    log_probs = torch.zeros(batch_size, NUM_CLASSES, device=device)
    for y in range(NUM_CLASSES):
        labels = torch.full((batch_size,), y, device=device, dtype=torch.long)
        logits = model(model_input, labels)
        # Use reduction='none' to get per-sample losses
        loss = discretized_mix_logistic_loss(model_input, logits)
        log_probs[:, y] = -loss    # Shape: [batch_size, NUM_CLASSES]
    _, answer = torch.max(log_probs, dim=1) 
    
    

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=8, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

  
    #Replace the random classifier with your trained model
    # Load the trained model 
    model = PixelCNN(nr_resnet=5, nr_filters=120, input_channels=3, conditional=True, nr_logistic_mix=10, condition_strength='mild')  # Match training params
     
  
    
    model = model.to(device)

    
    #Save the model to this path
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
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        