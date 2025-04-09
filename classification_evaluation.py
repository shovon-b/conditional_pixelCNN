'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils2 import *
from model2 import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    # batch_size,_,_,_=model_input.shape    
    # labels=torch.arange(NUM_CLASSES, device= device)
    # sample_op = lambda x : sample_from_discretized_mix_logistic(x, 5)
    # obs = (3,32,32)
    # prob_con=torch.tensor([], device= device)
    # for img_label in labels:
    #     print("sampling...")
    #     sample_t = sample(model, batch_size, obs, sample_op, img_label)
    #     sample_t = rescaling_inv(sample_t)
    #     temp = sample_t.contiguous().view(batch_size,-1)
    #     log_prob_con = torch.sum(torch.log(temp), dim = 1, keepdim = True)
    #     prob_con = torch.cat((prob_con, torch.exp(log_prob_con)), dim = 1)
    
    # con_prob = prob_con/torch.sum(prob_con, dim = 1, keepdim = True)    
    # _ , answer = torch.max(con_prob, dim =1)

    # batch_size = model_input.shape[0]
    # log_probs = torch.zeros(batch_size, NUM_CLASSES, device=device)
    # for y in range(NUM_CLASSES):
    #     labels = torch.full((batch_size,), y, device=device, dtype=torch.long)
    #     logits = model(model_input, labels)
    #     # Use reduction='none' to get per-sample losses
    #     loss = discretized_mix_logistic_loss(model_input, logits)
    #     log_probs[:, y] = -loss    # Shape: [batch_size, NUM_CLASSES]
    # _, answer = torch.max(log_probs, dim=1) 
    
    batch_size = model_input.shape[0]
    sample_op = lambda x: sample_from_discretized_mix_logistic(x, 5)
    obs = (3, 32, 32)
    
    # Vectorize sampling across all classes at once
    labels = torch.arange(NUM_CLASSES, device=device).repeat(batch_size, 1)  # Shape: [batch_size, num_classes]
    # Expand model_input to match the number of classes
    model_input_expanded = model_input.repeat_interleave(NUM_CLASSES, dim=0)  # Shape: [batch_size * num_classes, 3, 32, 32]
    labels_flat = labels.reshape(-1)  # Shape: [batch_size * num_classes]
    
    # Single sampling pass instead of per-class sampling
    print("sampling...")
    samples = sample(model, batch_size * NUM_CLASSES, obs, sample_op, labels_flat)
    samples = rescaling_inv(samples)  # Shape: [batch_size * num_classes, 3, 32, 32]
    
    # Reshape and compute probabilities
    samples = samples.view(batch_size, NUM_CLASSES, -1)  # Shape: [batch_size, num_classes, 3072]
    log_probs = torch.sum(torch.log(samples + 1e-10), dim=2)  # Shape: [batch_size, num_classes], added small epsilon to avoid log(0)
    
    # Softmax to get probabilities and get predictions
    probs = torch.softmax(log_probs, dim=1)
    predictions = torch.argmax(probs, dim=1)  # Shape: [batch_size]
    
    return predictions
    
  
        
# End of your code

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
                        default=32, help='Batch size for inference')
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

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5)    
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/pcnn_cpen455_load_model_59.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        