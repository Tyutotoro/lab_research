import numpy as np
import omnipose
from omnipose.gpu import use_gpu
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES
from cellpose_omni import io, transforms
from omnipose.utils import normalize99

import time
import tifffile
import cv2
from cellpose_omni import plot

def omnipose_model_eval():
    #でーたよみこみ
    omnidir = '/home/nakajima/work/Ecoli/code/Unet/data/64_16'
    basedir = os.path.join(omnidir,'test_manual2','image')
    # input_dir = '/home/nakajima/work/Ecoli/data/nd049_label_image/'
    files = io.get_image_files(basedir)
    # files = io.get_image_files(input_dir)
    use_GPU = use_gpu()
    imgs = [io.imread(f) for f in files]
    nimg = len(imgs)
    print(MODEL_NAMES)
    model_name = 'bact_phase_omni'
    model = models.CellposeModel(gpu=use_GPU, model_type=model_name)
    chans = [0,0] #this means segment based on first channel, no second channel 

    n = [-1] # make a list of integers to select which images you want to segment
    n = range(nimg) # or just segment them all 

    # define parameters
    params = {'channels':chans, # always define this with the model
            'rescale': None, # upscale or downscale your images, None = no rescaling 
            'mask_threshold': -2, # erode or dilate masks with higher or lower values between -5 and 5 
            'flow_threshold': 0, # default is .4, but only needed if there are spurious masks to clean up; slows down output
            'transparency': True, # transparency in flow output
            'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
            'cluster': True, # use DBSCAN clustering
            'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
            'verbose': False, # turn on if you want to see more output 
            'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed 
            'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
            'augment': False, # Can optionally rotate the image and average network outputs, usually not needed 
            'affinity_seg': False, # new feature, stay tuned...
            }

    tic = time.time() 
    masks, flows, styles = model.eval([imgs[i] for i in n],**params)

    net_time = time.time() - tic
    masks_np = np.array(masks)
    print('total segmentation time: {}s'.format(net_time))
    save_path = './result/S1_C1_T0_64_255/'
    for i, image in enumerate(masks_np):
            tifffile.imwrite(save_path + f"{i:05d}.tif", (image * (255/(np.max(image)))).astype(np.uint8))
            # print(image.shape)
    return masks, flows, styles

def omnipose_model_train():
    # モデルの初期化
    use_GPU = use_gpu()
    model_name = 'bact_phase_omni'
    model = models.CellposeModel(gpu=use_GPU, model_type=model_name)

    # データ読み込み
    images, masks,links, image_names, test_images, test_labels, test_links, image_names_test = io.load_train_test_data('./data/scaleimage256_removed0', mask_filter='_masks')
    print(images[0])
    print(type(images[0]))
    print(len(masks))
    print(type(masks[0]))
    train_links_list = [() for _ in range(len(images))]
    # 学習の実行
    model.train(train_data=images, train_labels=masks, train_links=train_links_list, train_files=None, 
            test_data=None, test_labels=None, test_links=None, test_files=None,
            channels=None, channel_axis=0, normalize=True, 
            save_path=None, save_every=100, save_each=False,
            learning_rate=0.0001, n_epochs=2, momentum=0.9, SGD=True,
            weight_decay=0.00001, batch_size=8, dataloader=False, num_workers=0, nimg_per_epoch=None,
            rescale=True, min_train_masks=5, netstr=None, tyx=None, timing=False, do_autocast=False,
            affinity_field=False)
        #  (images, masks, n_epochs=10, learning_rate=0.001)
if __name__ == '__main__':
    omnipose_model_train()
    omnipose_model_eval()

# for idx,i in enumerate(n):

#     maski = masks[idx] # get masks
#     bdi = flows[idx][-1] # get boundaries
#     flowi = flows[idx][0] # get RGB flows 

#     # set up the output figure to better match the resolution of the images 
#     f = 9
#     szX = maski.shape[-1]/mpl.rcParams['figure.dpi']*f
#     szY = maski.shape[-2]/mpl.rcParams['figure.dpi']*f
#     fig = plt.figure(figsize=(szY,szX*4), facecolor=[0]*4, frameon=False)
    
#     plot.show_segmentation(fig, omnipose.utils.normalize99(imgs[i]), 
#                         maski, flowi, bdi, channels=chans, omni=True,
#                         interpolation=None)

#     plt.tight_layout()
#     plt.show()