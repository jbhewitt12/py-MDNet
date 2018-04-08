# py-MDNet

by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

## ----------------------------------------------------------
## Modifications made by jbhewitt12
- ran all .py files through [2to3](https://docs.python.org/2/library/2to3.html#using-2to3) 
- added dim=0 paramater to log_softmax() pytorch function to fix error

## Errors encountered by jbhewitt12
After opening command prompt in the /tracking folder and running "python run_tracker.py -s DragonBaby -d":

```
../modules\model.py:144: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  pos_loss = -F.log_softmax(pos_score)[:,1]
../modules\model.py:145: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  neg_loss = -F.log_softmax(neg_score)[:,0]
Traceback (most recent call last):
  File "run_tracker.py", line 329, in <module>
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)
  File "run_tracker.py", line 190, in run_mdnet
    im = ax.imshow(image, aspect='normal')
  File "C:\Users\Josh\Anaconda3\lib\site-packages\matplotlib\__init__.py", line 1717, in inner
    return func(ax, *args, **kwargs)
  File "C:\Users\Josh\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py", line 5126, in imshow
    self.set_aspect(aspect)
  File "C:\Users\Josh\Anaconda3\lib\site-packages\matplotlib\axes\_base.py", line 1292, in set_aspect
    self._aspect = float(aspect)  # raise ValueError if necessary
ValueError: could not convert string to float: 'normal'
```

I added dim=0 paramater to log_softmax() pytorch function in an attempt to fix the error but it only got rid of the UserWarning part of the error. Not sure how to proceed. 

## ----------------------------------------------------------

## Introduction
Python (PyTorch) implementation of MDNet tracker, which is ~2x faster than the original matlab implementation. 
#### [[Project]](http://cvlab.postech.ac.kr/research/mdnet/) [[Paper]](https://arxiv.org/abs/1510.07945) [[Matlab code]](https://github.com/HyeonseobNam/MDNet)

If you're using this code for your research, please cite:

	@InProceedings{nam2016mdnet,
	author = {Nam, Hyeonseob and Han, Bohyung},
	title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2016}
	}
 
## Prerequisites
- python 2.7
- [PyTorch](http://pytorch.org/) and its dependencies 

## Usage

### Tracking
```bash
 cd tracking
 python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python run_tracker.py -s [seq name]```
   - ```python run_tracker.py -j [json path]```
 
### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
``` bash
 cd pretrain
 python prepro_data.py
 python train_mdnet.py
```
