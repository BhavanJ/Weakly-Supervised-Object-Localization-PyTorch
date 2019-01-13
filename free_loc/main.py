import argparse
import os
import shutil
import time
import sys
sys.path.insert(0,'/home/spurushw/reps/hw-wsddn-sol/faster_rcnn')
sys.path.insert(0,'/home/bjasani/Desktop/CMU_HW/VLR/HW2/hw2-release/code/faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pdb
import visdom

import matplotlib.pyplot as plt

from datasets.factory import get_imdb
from custom import *
from logger import Logger

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


#EXPERIMENT_NAME = 'task_5_version_5'
EXPERIMENT_NAME = 'task_1_5_version_1'
VISUAL_IM = True
VIS_PORT_NO = '5001' 
#python -m visdom.server -port 5000


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', default='localizer_alexnet')
#parser.add_argument('--arch', default='localizer_alexnet_robust')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--eval-freq', default=2, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')

parser.add_argument('--pretrained', default=True, type=bool,
                    help='use pre-trained model')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')

parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

parser.add_argument('--vis',action='store_true')

best_prec1 = 0


vis = visdom.Visdom(port=VIS_PORT_NO)

class_id_to_name = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 
'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 
'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    #pdb.set_trace()

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch=='localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
        #model = localizer_alexnet(pretrained=True)
    elif args.arch=='localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
        #model = localizer_alexnet_robust(pretrained=True)

    print(model)

    #pdb.set_trace()

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()


    # TODO:
    # define loss function (criterion) and optimizer

    #criterion = nn.SoftMarginLoss().cuda()                      ######VERIFY
    criterion = nn.MultiLabelSoftMarginLoss().cuda()     ######VERIFY
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py

    

    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512,512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    ###########VERIFY TODO just for debugging
    #epoch = 1
    #train(train_loader, model, criterion, optimizer, epoch)    
    

    #validate(val_loader, model, criterion)
    if args.evaluate:
        validate(val_loader, model, criterion,epoch,logger)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()

    log_dir = '/home/bjasani/Desktop/CMU_HW/VLR/HW2/hw2-release/code/tf_logs/free_loc/new'
    logger = Logger(log_dir, name = EXPERIMENT_NAME)
    


    #pdb.set_trace()


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger)

        # evaluate on validation set
        if epoch%args.eval_freq==0 or epoch==args.epochs-1:
            m1, m2 = validate(val_loader, model, criterion,epoch,logger)
            score = m1*m2
            # remember best prec@1 and save checkpoint
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)




    pdb.set_trace()
        

#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    avg_m3 = AverageMeter()
    avg_m4 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        
        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output

       

        ####output.shape --> torch.Size([32, 20, 29, 29])
        output_heatmap = model(input_var)

        #size = (N=32,C=1,H,W)    
        upsampled_heatmap = F.upsample(input= output_heatmap, size = (input.shape[2],input.shape[3]) , mode = 'bilinear')    
        #torch.Size([32, 20, 512, 512])

        output = F.max_pool2d(output_heatmap, kernel_size = (output_heatmap.shape[2],output_heatmap.shape[3]) )
        
        
        #loss = 0
        loss = criterion(output[:,:,0,0], target_var )
            
        imoutput = output[:,:,0,0] ##########VERIFY WE INSERTED

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        m3 = metric3(imoutput.data, target)
        m4 = metric4(imoutput.data, target)


        #pdb.set_trace()
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))
        avg_m3.update(m3[0], input.size(0))
        avg_m4.update(m4[0], input.size(0))


        
        # TODO: 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, avg_m1=avg_m1,
                   avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        logger.scalar_summary(tag = 'train/loss',    value = loss,  step = (epoch*len(train_loader)) + i )
        logger.scalar_summary(tag = 'train/metric1', value = m1[0], step = (epoch*len(train_loader)) + i )
        logger.scalar_summary(tag = 'train/metric2', value = m2[0], step = (epoch*len(train_loader)) + i )
        logger.scalar_summary(tag = 'train/metric3', value = m3[0], step = (epoch*len(train_loader)) + i )
        logger.scalar_summary(tag = 'train/metric4', value = m4[0], step = (epoch*len(train_loader)) + i )
        logger.model_param_histo_summary(model = model, step = (epoch*len(train_loader)) + i )

        c_map = plt.get_cmap('jet')

        #pdb.set_trace()    

        if (i == 0 or i == len(train_loader)/3 or i == 2*len(train_loader)/3 or i == (len(train_loader)-1)) and VISUAL_IM == True:

            for batchindx in range(input.shape[0]/8):

                if epoch == 0 or epoch == (args.epochs-1):
                #if epoch%2 ==0:
                    #pdb.set_trace()
                    title_vis_i = 'epoch_' + str(epoch) + '/iteration_' + str(i) + '/batch_index_' + str(batchindx)
                    plot_image_vis = input[batchindx,:,:,:]
                    plot_image_vis = (plot_image_vis - plot_image_vis.min())/(plot_image_vis.max()-plot_image_vis.min())
                    vis.image((plot_image_vis), opts=dict(caption=title_vis_i))

                image_no = batchindx
                logger.image_summary(tag = 'train/image_heatmap/epoch_' + str(epoch) + '/iter_' + str(i) + '/batch_indx_' + str(image_no), images = input[image_no:image_no+1,:,:,:], step = (epoch*len(train_loader)) + i)
                heatmap_of_image = upsampled_heatmap[image_no,:,:,:]
                gt_labels_image = target[image_no,:]                

                for clsid in range(heatmap_of_image.shape[0]):
                    if gt_labels_image[clsid] == 1:

                        heatmap_per_class = heatmap_of_image[clsid:clsid+1,:,:]
                        heatmap_per_class = (heatmap_per_class - heatmap_per_class.min()) / (heatmap_per_class.max() - heatmap_per_class.min())                
                        plot_heatmap_tf = c_map(heatmap_per_class.data.cpu().numpy()[0,:,:])    
                        plot_heatmap_tf = np.delete(plot_heatmap_tf, 3, 2)
                        plot_heatmap_tf = plot_heatmap_tf.transpose(2,0,1)
                        plot_heatmap_tf = np.reshape(plot_heatmap_tf,(1,3,512,512))
                        #logger.image_summary(tag = 'train/heatmaps/image_' + str(image_no) + '/gtclass_' + str(clsid) , images = plot_heatmap_tf, step = (epoch*len(train_loader)) + i)
                        logger.image_summary(tag = 'train/image_heatmap/epoch_' + str(epoch) + '/iter_' + str(i) + '/batch_indx_' + str(image_no)+ '/gtclass_' + str(clsid), images = plot_heatmap_tf, step = (epoch*len(train_loader)) + i)


                        if epoch == 0 or epoch == (args.epochs-1):
                        #if epoch%2 ==0:
                            #plot_heatmap_vis = np.transpose(plot_heatmap_tf[0,:,:,:], (2, 0,1))    
                            plot_heatmap_vis = plot_heatmap_tf[0,:,:,:]
                            title_vis_h = 'epoch_' + str(epoch) + '/iteration_' + str(i) + '/batch_index_' + str(image_no) + '/heatmap_class_name/' + class_id_to_name[clsid]                  
                            vis.image((plot_heatmap_vis), opts=dict(caption=title_vis_h))








def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    avg_m3 = AverageMeter()
    avg_m4 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output

        output_heatmap = model(input_var)
        upsampled_heatmap = F.upsample(input= output_heatmap, size = (input.shape[2],input.shape[3]) , mode = 'bilinear')    
        output = F.max_pool2d(output_heatmap, kernel_size = (output_heatmap.shape[2],output_heatmap.shape[3]) )
        loss = criterion(output[:,:,0,0], target_var )

        #output = nn.functional.max_pool2d(output, kernel_size = (output.shape[1],output.shape[2]) )
        #loss = 0
        #for cl in range(20):
        #    loss += criterion(output[cl], 1 if target_var[cl] ==1 else -1)        



        imoutput = output[:,:,0,0] ##########VERIFY WE INSERTED
        

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        m3 = metric3(imoutput.data, target)
        m4 = metric4(imoutput.data, target)
        
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))
        avg_m3.update(m3[0], input.size(0))
        avg_m4.update(m4[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   avg_m1=avg_m1, avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        img_ctr=0
        c_map = plt.get_cmap('jet')

        if epoch == (args.epochs-1) and i%5 == 0:
            
            image_no = 1
            title_vis_i = 'Validation_images_after_training/' + str(img_ctr) #'epoch_' + str(epoch) + '/iteration_' + str(i) + '/batch_index_' + str(batchindx)
            plot_image_vis = input[image_no,:,:,:]
            plot_image_vis = (plot_image_vis - plot_image_vis.min())/(plot_image_vis.max()-plot_image_vis.min())
            vis.image((plot_image_vis), opts=dict(caption=title_vis_i))

            heatmap_of_image = upsampled_heatmap[image_no,:,:,:]
            gt_labels_image = target[image_no,:]         

            for clsid in range(heatmap_of_image.shape[0]):
                if gt_labels_image[clsid] == 1:

                    heatmap_per_class = heatmap_of_image[clsid:clsid+1,:,:]
                    heatmap_per_class = (heatmap_per_class - heatmap_per_class.min()) / (heatmap_per_class.max() - heatmap_per_class.min())                
                    plot_heatmap_tf = c_map(heatmap_per_class.data.cpu().numpy()[0,:,:])    
                    plot_heatmap_tf = np.delete(plot_heatmap_tf, 3, 2)
                    plot_heatmap_tf = plot_heatmap_tf.transpose(2,0,1)
                    plot_heatmap_tf = np.reshape(plot_heatmap_tf,(1,3,input.shape[2],input.shape[3]))
                    plot_heatmap_vis = plot_heatmap_tf[0,:,:,:]
                    title_vis_h = 'Validation_heatmap_after_training/' + str(img_ctr) + '/heatmap_class_name/' + class_id_to_name[clsid]
                    vis.image((plot_heatmap_vis), opts=dict(caption=title_vis_h))


            img_ctr +=1    



    logger.scalar_summary(tag = 'validate/metric1', value = avg_m1.avg, step = epoch  )
    logger.scalar_summary(tag = 'validate/metric2', value = avg_m2.avg, step = epoch  )
    logger.scalar_summary(tag = 'validate/metric3', value = avg_m3.avg, step = epoch  )
    logger.scalar_summary(tag = 'validate/metric4', value = avg_m4.avg, step = epoch  )

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target, threshold=0.5):
    # TODO: Ignore for now - proceed till instructed

    ap = sklearn.metrics.average_precision_score(target.cpu().numpy(), output.cpu().numpy(), average='micro')
    return [ap]

    # for iii in range( output.shape[0] ):
    #     output[iii,:]  = (output[iii,:] - output[iii,:].min() ) / (output[iii,:].max() - output[iii,:].min())
    #     output[iii,:] = output[iii,:] > threshold
    
    # ap = sklearn.metrics.average_precision_score(target.cpu().numpy(), output.cpu().numpy(), average='micro')
    # return [ap]

def metric2(output, target, threshold=0.5):
    # TODO: Ignore for now - proceed till instructed

    #pdb.set_trace()
    output_sigmoid = F.sigmoid(output)
    output_binary = output_sigmoid>threshold
    sig_recall = sklearn.metrics.recall_score(np.int32(target.cpu().numpy()), np.int32(output_binary.cpu().numpy()), average='micro')
    return [sig_recall]

    # for iii in range( output.shape[0] ):
    #     output[iii,:]  = (output[iii,:] - output[iii,:].min() ) / (output[iii,:].max() - output[iii,:].min())
    #     output[iii,:] = output[iii,:] > threshold
    
    # recall = sklearn.metrics.recall_score(target.cpu().numpy(), output.cpu().numpy(), average='micro')
    # return [recall]

def metric3(output, target, threshold=0.5):

    output_sigmoid = F.sigmoid(output)
    output_binary = output_sigmoid>threshold
    sig_ap = sklearn.metrics.average_precision_score(target.cpu().numpy(), output_binary.cpu().numpy(), average='micro')
    return [sig_ap]

def metric4(output, target, threshold=0.8):

    output_sigmoid = F.sigmoid(output)
    output_binary = output_sigmoid>threshold
    sig_recall = sklearn.metrics.recall_score(np.int32(target.cpu().numpy()), np.int32(output_binary.cpu().numpy()), average='micro')
    return [sig_recall]



if __name__ == '__main__':
    main()




