import os
from torch.utils.data import Dataset
from PIL import Image



class myDataset(Dataset): 
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_path_list = []

        for dir in os.listdir(root):
            dir_path = os.path.join(root, dir)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                self.image_path_list.append(file_path)

    def __getitem__(self,index):
        image_path = self.image_path_list[index]

        img = Image.open(image_path).convert("RGB")

        return self.transform(img),"yira" 

    def __len__(self):
        return len(self.image_path_list)
