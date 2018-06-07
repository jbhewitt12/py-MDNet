import os
import json
import numpy as np

def gen_config(args, i):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = '../dataset/'
        save_home = '../result_fig'
        result_home = '../result'
        
        seq_list = [
        "david",
        "diving",
        "drunk",
        "hand1",
        "jogging",
        "polarbear",
        "skating",
        "sunshade",
        "surfing",
        "torus",
        "trellis",
        "woman"
        ]
        print("**SEQUENCE:")
        print(seq_list[i])
        seq_name = "vot2014/"+seq_list[i]
        img_dir = os.path.join(seq_home, seq_name)
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')


        img_list = os.listdir(img_dir)
        img_list.sort()
        newlist = []
        for names in img_list:
            if names.endswith(".jpg"):
                newlist.append(names)
        img_list = newlist
        # print('newlist: ')
        # print(newlist)
        # print('img_list before: ')
        # print(img_list)
        img_list = [os.path.join(img_dir,x) for x in img_list]
        # print('img_list: ')
        # print(img_list)


        gt = np.loadtxt(gt_path,delimiter=',')
        print('no of groundtruths:')
        print(len(gt))
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

#                                 python run_tracker.py -s vot2014/bicycle -d

# def convert_groundtruth(gt):
#     print("converting gt")
#     print('gt: ')
#     print(gt)
#     # print(len(gt))
#     truths = []
    

#     for row in gt:
#         box = []
#         box.append(row[2])
#         box.append(row[3])
#         box.append(abs(row[2] - row[4]))
#         box.append(abs(row[1] - row[3]))
#         truths.append(box)

#     truths = np.asarray(truths)
#     print('truths:')
#     print(truths)
#     return truths

def convert_groundtruth(gt):
    # print("converting gt")
    # print('gt: ')
    # print(gt)
    # print(len(gt))
    truths = []
    

    for row in gt:
        xvals = [row[0],row[2],row[4],row[6]]
        yvals = [row[1],row[3],row[5],row[7]]
        top = max(yvals)
        bottom = min(yvals)
        left = min(xvals)
        right = max(xvals)
        box = []
        # print('------********')
        # print(xvals)
        # print(yvals)
        # print(top)
        # print(bottom)
        # print(left)
        # print(right)
        box.append(left)
        box.append(bottom)
        box.append(abs(right - left))
        box.append(abs(top - bottom))
        truths.append(box)

    truths = np.asarray(truths)
    # print('truths:')
    # print(truths)
    return truths


