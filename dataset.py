import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize
from bidict import bidict
import pandas as pd

# Your existing lambdas and bidict
rescaling = lambda x: (x - 0.5) * 2.
rescaling_inv = lambda x: 0.5 * x + 0.5
replicate_color_channel = lambda x: x.repeat(3, 1, 1)

my_bidict = bidict({'Class0': 0, 'Class1': 1, 'Class2': 2, 'Class3': 3})

class CPEN455Dataset(Dataset):
    def __init__(self, root_dir='/kaggle/working/data', mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            mode (string): 'train', 'validation', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []  # Preloaded images
        self.categories = []  # Preloaded category names
        
        # Load CSV and populate samples
        csv_path = os.path.join(self.root_dir, mode + '.csv')
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        samples = [(os.path.join(self.root_dir, path), label) for path, label in df.itertuples(index=False, name=None)]
        
        # Preload all images into memory
        for img_path, category in samples:
            image = read_image(img_path).type(torch.float32) / 255.  # Normalize to [0, 1]
            if image.shape[0] == 1:
                image = replicate_color_channel(image)
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
            category_name = my_bidict.inverse[category] if category in my_bidict.values() else "Unknown"
            self.categories.append(category_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.categories[idx]

    def get_all_images(self, label):
        return [img for img, cat in zip(self.images, self.categories) if cat == label]

# Your existing show_images function (unchanged)
def show_images(images, categories, mode: str):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, image in enumerate(images):
        axs[i].imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axs[i].set_title(f"Category: {categories[i]}")
        axs[i].axis('off')
    plt.savefig(mode + '_test.png')

# Integration with your training setup
if __name__ == '__main__':
    transform_32 = Compose([
        Resize((32, 32)),  # Resize images to 32 * 32
        rescaling
    ])
    
    # Example usage with your config
    dataset = CPEN455Dataset(root_dir='/kaggle/working/data', mode='train', transform=transform_32)
    data_loader = DataLoader(dataset, batch_size=48, num_workers=0, pin_memory=True, drop_last=True)
    
    # Test one batch
    from tqdm import tqdm
    for images, categories in tqdm(data_loader):
        print(images.shape, categories)
        break
