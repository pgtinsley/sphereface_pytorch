from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import os, sys, cv2, random, datetime
import argparse
import numpy as np

from PIL import Image

from dataset import ImageDataset

import net_sphere

import glob
import pickle

from facenet_pytorch import MTCNN


parser = argparse.ArgumentParser(description='PyTorch SphereFace')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model','-m', default='./model/SphereFace.pth', type=str)
parser.add_argument('--dir', '-d', required=True, type=str)
parser.add_argument('--embeddings', '-e', required=True, type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere, args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

def batch_save(i, subset_i, subset_fname):

    imglist = []
    for j, fname in zip(subset_i, subset_fname):
        img = np.array(Image.open(fname))
        imglist.append(img)

    for f in range(len(imglist)):
        imglist[f] = imglist[f].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[f] = (imglist[f]-127.5)/128.0

    stack = np.vstack(imglist)
    for_net = Variable(torch.from_numpy(stack).float()).cuda()
    output = net(for_net)

    encodings = {}
    for k, (fname, enc) in enumerate(zip(subset_fname, output.data)):
        encodings[k] = {
            'fname': fname.split('/')[-1],
            'enc': enc
        }
    
    with open('./tmp_pkls/encodings_{}to{}.pkl'.format(i,i+25), 'wb') as f:
        pickle.dump(encodings, f)



mtcnn = MTCNN(device='cuda:0')

dir = [args.dir if args.dir[-1]=='/' else args.dir+'/'][0]

img_fnames_orig = glob.glob(dir+'*.png') + glob.glob(dir+'*.jpg') + glob.glob(dir+'*.jpeg')

import shutil

if os.path.isdir('./tmp_dir_160x160/'):
    shutil.rmtree('./tmp_dir_160x160/')
os.mkdir('./tmp_dir_160x160/')

print('Writing to 160x160...')
for fname in img_fnames_orig:
    img = cv2.imread(fname)
    img_cropped = mtcnn(img, save_path='./tmp_dir_160x160/'+fname.split('/')[-1])

if os.path.isdir('./tmp_dir_96x112/'):
    shutil.rmtree('./tmp_dir_96x112/')
os.mkdir('./tmp_dir_96x112/')

img_fnames_160x160 = glob.glob('./tmp_dir_160x160/*.png') + glob.glob(dir+'./tmp_dir_160x160/*.jpg') + glob.glob(dir+'./tmp_dir_160x160/*.jpeg')

print('Writing to 96x112...')
for fname in img_fnames_160x160:
    img_160x160 = cv2.imread(fname)
    img_96x112 = cv2.resize(img_160x160, (96,112))
    cv2.imwrite('./tmp_dir_96x112/'+fname.split('/')[-1], img_96x112)

img_fnames = glob.glob('./tmp_dir_96x112/*.png') + glob.glob(dir+'./tmp_dir_96x112/*.jpg') + glob.glob(dir+'./tmp_dir_96x112/*.jpeg')

if os.path.isdir('./tmp_pkls/'):
    shutil.rmtree('./tmp_pkls/')
os.mkdir('./tmp_pkls/')

print('Beginning batch saving...')
for i in range(0, len(img_fnames), 25):
    print(i)
    if i+25 > len(img_fnames): # tail end
        subset_i = [j for j in range(i, len(img_fnames))]
        subset_fname = img_fnames[i: len(img_fnames)]
        batch_save(i, subset_i, subset_fname)
    else:
        subset_i = [j for j in range(i, i+25)]
        subset_fname = img_fnames[i: i+25]
        batch_save(i, subset_i, subset_fname)

pkl_fnames = glob.glob('./tmp_pkls/*.pkl')

combined = {}
for pkl in pkl_fnames:
    with open(pkl, 'rb') as f:
        tmp = pickle.load(f)
        for i, d in tmp.items():
            combined[d['fname']] = d['enc']

with open(args.embeddings, 'wb') as f2:
    pickle.dump(combined, f2)

print('Results saved to {}'.format(args.embeddings))
exit()

            
    


# os.mkdir('./tmp_dir_160x160/')


# for fname in img_fnames[0:10]:
#     img = np.array(Image.open(fname))
#     det = mtcnn(img).numpy()
#     det_rs = cv2.resize(det, (120,96))
#     print(det_rs.shape)
# 
#     for_net = Variable(torch.from_numpy(det_rs).float()).cuda()
#     output = net(for_net)
#     print(output.data)

# for i in range(0, len(img_fnames), 25):
#     print(i)
#     if i+25 > len(img_fnames): # tail end
#         subset_i = [j for j in range(i, len(img_fnames))]
#         subset_fname = img_fnames[i: len(img_fnames)]
#         batch_save(i, subset_i, subset_fname)
#     else:
#         subset_i = [j for j in range(i, i+25)]
#         subset_fname = img_fnames[i: i+25]
#         batch_save(i, subset_i, subset_fname)


