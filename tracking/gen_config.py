import os
import json
import numpy as np

def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = '../dataset/'
        save_home = '../result_fig'
        result_home = '../result'
        
        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name)
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')


        img_list = os.listdir(img_dir)
        img_list.sort()
        newlist = []
        for names in img_list:
            if names.endswith(".jpg"):
                newlist.append(names)
        img_list = newlist
        img_list = [os.path.join(img_dir,x) for x in img_list]

        gt = np.loadtxt(gt_path,delimiter=',')
        if(len(gt[0]) == 8):
            gt = convert_groundtruth(gt)

        init_bbox = gt[0]
        

        
        savefig_dir = os.path.join(save_home,seq_name)
        result_dir = os.path.join(result_home,seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir,'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path

def convert_groundtruth(gt): #This function converts from VOT2014+ groundtruth format (x1,y1,x2,y2,x3,y3,x4,y4) to VOT2013 groundtruth format (x1,y1,width,height)
    
    truths = []
    
    for row in gt:
        xvals = [row[0],row[2],row[4],row[6]]
        yvals = [row[1],row[3],row[5],row[7]]
        top = max(yvals)
        bottom = min(yvals)
        left = min(xvals)
        right = max(xvals)
        box = []
        box.append(left)
        box.append(bottom)
        box.append(abs(right - left))
        box.append(abs(top - bottom))
        truths.append(box)

    truths = np.asarray(truths)
    return truths


