import os
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Mixed Precision with Apex and Monitoring with Wandb
import wandb
from apex import amp
from apex.optimizers import FusedAdam

# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
#wandb.login()

#GPU using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    os.makedirs("./saved")
except FileExistsError:
    # directory already exists
    pass

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

#Hyperparameters
config = dict(
    TRAIN_CSV = "../data/train.csv",
    TEST_CSV = "../data/test.csv",
    IMAGE_PATH= "../data/images",
    VOCAB = "labels.json",
    saved_path="./saved/resnet18_ddp.pt",
    lr=0.001, 
    EPOCHS = 50,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    SEED = 42,
    device=device,
    pin_memory=True,
    num_workers=8,
    USE_AMP = True,
    channels_last=True,
    distributed = True,
    world_size=4)


#Initiate the Project and Entity
wandb.init(project="pytorch-lab", config=config, entity="amit-k203")
# access all HPs through wandb.config, so logging matches execution!
config = wandb.config

if config.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)       
    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')


#Pytorch Reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

#Visualize data
train_df = pd.read_csv(config.TRAIN_CSV)
test_df = pd.read_csv(config.TEST_CSV)
f = open(config.VOCAB)
vocab = json.load(f)

df_fnames = train_df["image_id"].append(test_df["image_id"],ignore_index=True).tolist()
def create_fname(path,extension):
    def add_extension(fname):
        return os.path.join(path,fname)+extension
    return add_extension

jpeg_extension_creator = create_fname(config.IMAGE_PATH,".jpg")
train_df["image_id"] = train_df["image_id"].apply(jpeg_extension_creator)
test_df["image_id"] = test_df["image_id"].apply(jpeg_extension_creator)
for label in vocab:
    train_df.loc[train_df[label] == 1, "label" ] = vocab[label] 
train_df["label"] = train_df["label"].astype(int)

train_df_X, valid_df_X, train_df_y, valid_df_y = train_test_split(
                                                    train_df["image_id"],
                                                    train_df["label"], 
                                                    test_size=config.TRAIN_VALID_SPLIT, 
                                                    random_state=0)

train_df_split = pd.DataFrame(data={"image_id": train_df_X, "label": train_df_y})
train_df_split.to_csv("../data/train_split.csv", sep=',',index=False)

valid_df_split = pd.DataFrame(data={"image_id": valid_df_X, "label": valid_df_y})
valid_df_split.to_csv("../data/val_split.csv", sep=',',index=False)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config.IMAGE_SIZE,config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



class PlantPathologyDataset(Dataset):
    def __init__(self,x,y,vocab,transforms):
        self.x = x # File Path in CSV
        self.y = y # Label in CSV
        self.vocab = vocab # Dictionary
        self.transforms = transforms
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx): #File Name --> Preprocessed 3-D Tensor
        fname = self.x.iloc[idx]        
        label = self.y.iloc[idx]
        image = Image.open(fname)
        
        if self.transforms:
            image = self.transforms(image)

        return image, label #[3,224,224], [0-3] 
    

train_ds = PlantPathologyDataset(train_df_X, 
                                 train_df_y, 
                                 vocab,
                                 data_transforms["train"])
valid_ds = PlantPathologyDataset(valid_df_X, 
                                 valid_df_y,
                                 vocab,
                                 data_transforms["val"])

################################################################
train_sampler = DistributedSampler(
    train_ds,
    num_replicas=config.world_size,
    rank=args.local_rank
)
################################################################


#Dataloader -> Set pin_memory=True and num_workers=8
train_dl = DataLoader(train_ds,
                      batch_size=config.BATCH_SIZE,
                      shuffle=False,
                      num_workers=config.num_workers,
                      pin_memory=config.pin_memory,
                      sampler=train_sampler)

valid_dl = DataLoader(valid_ds,
                      batch_size=config.BATCH_SIZE,
                      shuffle=False,
                      num_workers=config.num_workers,
                      pin_memory=config.pin_memory)


model = models.resnet18(pretrained=True)

#Modify the classifier for agriculture data
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,512),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(512,4))


#Channel Last Optimization in Model
if config.channels_last:
    model = model.to(config.device, memory_format=torch.channels_last) #CHW --> #HWC
else:
    model = model.to(config.device)
    
if config.USE_AMP:
    optimizer = FusedAdam(model.parameters(), config.lr)
    model,optimizer = amp.initialize(model, optimizer, opt_level="O2") #O0/O1/O2
else:
    optimizer = optim.Adam(model.parameters(),lr=config.lr)
    
if config.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    # model = DistributedDataParallel(model)
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added configs:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    
# Loss Function
criterion = nn.CrossEntropyLoss()

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.

def train_model(model,criterion,optimizer,num_epochs=10):
    ############################################################
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    ############################################################

    since = time.time()                                            
    batch_ct = 0
    example_ct = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        #Training
        model.train()
        for x,y in train_dl: #BS=32 ([BS,3,224,224], [BS,4])            
            if config.channels_last:
                x = x.to(config.device, non_blocking=True, memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True) #CHW --> #HWC
            optimizer.zero_grad()
            train_logits = model(x) #Input = [BS,3,224,224] (Image) -- Model --> [BS,4] (Output Scores)
            _, train_preds = torch.max(train_logits, 1)
            train_loss = criterion(train_logits,y)
            
            ########################################################################
            if config.USE_AMP:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    loss=scaled_loss
            else:
                train_loss.backward() # Backpropagation this is where your W_gradient
                loss=train_loss

            optimizer.step() # W_new = W_old - LR * W_gradient 
            example_ct += len(x) 
            batch_ct += 1
            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
                
        #validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        with torch.no_grad():
            for x,y in valid_dl:
                if config.channels_last:
                    x = x.to(config.device, non_blocking=True, memory_format=torch.channels_last) #CHW --> #HWC
                else:
                    x = x.to(config.device, non_blocking=True)
                y = y.to(config.device, non_blocking=True) #CHW --> #HWC
                valid_logits = model(x)
                _, valid_preds = torch.max(valid_logits, 1)
                valid_loss = criterion(valid_logits,y)
                running_loss += valid_loss.item() * x.size(0)
                running_corrects += torch.sum(valid_preds == y.data)
                total += y.size(0)
                wandb.log({"test_accuracy": running_corrects / total})
            
        epoch_loss = running_loss / len(valid_ds)
        epoch_acc = running_corrects.double() / len(valid_ds)
        print("Validation Loss is {}".format(epoch_loss))
        print("Validation Accuracy is {}".format(epoch_acc.cpu()))

            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    torch.save(model.state_dict(), config.saved_path)

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
    
train_model(model, criterion, optimizer, num_epochs=config.EPOCHS)
    
    