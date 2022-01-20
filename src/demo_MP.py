from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fileinput import filename

from numpy.core.fromnumeric import size

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

# multiple thread 
import threading
from multiprocessing import Process

import copy

logger.setLevel(logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def demo(opt,deviceid):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceid)
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    print ('video' ,opt.input_video)  
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    # Save output frames to folder, for generating video with ffmpeg
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=True, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    # Convert output frames into video file, e.g. mp4
    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)

def file_name(file_dir):   
    videofiles=[]
    for root, dirs, files in os.walk(file_dir):  
        print('root',root) #当前目录路径  
        print('dirs',dirs) #当前路径下所有子目录  
        if size(files)<3 and size(files)>0:
            firts=root+'/'+files[0]
            print('files',firts) #当前路径下所有非目录子文件  
            videofiles.append(firts)
    return videofiles
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    path='../All_Data'
    videoPaths = file_name(path)
    print('files',videoPaths)
    opt = opts().init()
    process=[]
    i=0
    for inputvideopath in videoPaths:
        print('first thread',inputvideopath)
        newopt=copy.deepcopy(opt)
        newopt.input_video=inputvideopath
        #opt.task,'--load_model' '../models/fairmot_lite.pth','--input_video',opt.input_video
        p=Process(target=demo, args=(newopt,(i+1)/2))
        process.append(p)
        i=i+1

    for t in process:
        t.start()
    for t in process:
        t.join()



