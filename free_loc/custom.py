import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np
#from myutils import *
import pdb
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index

    classes = imdb._classes
    class_to_idx = imdb._class_to_ind

    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset

    #imdb.image_path_at(0)
    #'/home/bjasani/Desktop/CMU_HW/VLR/HW2/hw2-release/code/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'

    #imdb.gt_roidb()[0]['gt_classes']
    #array([9, 9, 9], dtype=int32)

    images = []
    annotations_all = imdb.gt_roidb()    

    for indx in range( len(annotations_all) ):

        diff_objects = []
        anot_at_index = annotations_all[indx]['gt_classes']

        for ctr in range(len(anot_at_index)):
            if anot_at_index[ctr] not in diff_objects:
                diff_objects.append(anot_at_index[ctr])        
        
        temp_tuple = (  imdb.image_path_at(indx), diff_objects )
        images.append(temp_tuple)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1)),
            )

    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)    
        x = self.classifier(x)
        return x




class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Ignore for now until instructed
        self.features  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout2d(p=0.5)
            )


    def forward(self, x):
        #TODO: Ignore for now until instructed
        x = self.features(x)    
        x = self.classifier(x)
        
        x =  F.dropout2d(x, p=0.5)
        x1 = F.avg_pool2d(x, kernel_size = 3 , stride  = 1, padding = 1 )
        x2 = F.avg_pool2d(x, kernel_size = 5,  stride  = 1, padding = 2 )
        x3 = F.avg_pool2d(x, kernel_size = 7,  stride  = 1, padding = 3 )
        x = (x + x1 + x2 + x3)/4.0

        return x



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)


def localizer_alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whether it is pretrained or
    #not
    #pdb.set_trace()

    if pretrained==True:
		#pretrained_dict = torch.load(model_urls['alexnet'])

        ####################NEW WAY###############
        ####alexnet_model = models.alexnet(pretrained=True)
        ####model.features.load_state_dict(alexnet_model.features.state_dict())
        ####return model

        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        #Initialzing all weights to Xavier
        model.apply(weights_init)
        
        our_model_dict = model.state_dict()
        print('\n\nUSING PRETRAINED MODEL\n\n')

		#kkk = 'features.10.bias'
		#our_model_dict[kkk] = pretrained_dict[kkk]
		#for k, v in pretrained_dict.items():
		#	print(k)
		#print('\n')	
		#for k,v in our_model_dict.items():
		#	print(k)	
		#for k,v in pretrained_dict_part.items():
		#	print(k)	

        pretrained_dict_part = {k: v for k, v in pretrained_dict.items() if k.startswith('features')}
        our_model_dict.update(pretrained_dict_part)

        model.load_state_dict(our_model_dict)

		###################################################
		 
	        
        #print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
    #else:
    #    #print("=> creating model '{}'".format(args.arch))
    #    #model = models.__dict__[args.arch]()
    #    pass
    
    return model

def localizer_alexnet_robust(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed

    if pretrained==True:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model.apply(weights_init)
        our_model_dict = model.state_dict()
        print('\n\nUSING PRETRAINED ROBUST MODEL\n\n')
        pretrained_dict_part = {k: v for k, v in pretrained_dict.items() if k.startswith('features')}
        our_model_dict.update(pretrained_dict_part)
        model.load_state_dict(our_model_dict)
    return model





class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write the rest of this function
        
        img = Image.open(self.imgs[index][0])
        target = np.zeros(len(self.classes))

        for j in self.imgs[index][1]:
            target[j-1] = 1                             #Verify, the clasess are labeled from 1 to 20 so -1


        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




##################################
'''
Pretrained model keys
features.0.weight
features.0.bias
features.3.weight
features.3.bias
features.6.weight
features.6.bias
features.8.weight
features.8.bias
features.10.weight
features.10.bias
classifier.1.weight
classifier.1.bias
classifier.4.weight
classifier.4.bias
classifier.6.weight
classifier.6.bias


Our model keys
features.0.weight
features.0.bias
features.3.weight
features.3.bias
features.6.weight
features.6.bias
features.8.weight
features.8.bias
features.10.weight
features.10.bias
classifier.0.weight
classifier.0.bias
classifier.2.weight
classifier.2.bias
classifier.4.weight
classifier.4.bias
'''