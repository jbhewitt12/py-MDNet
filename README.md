# py-MDNet

##Fork notes

This is a fork of [py-MDNet](https://github.com/HyeonseobNam/py-MDNet) by [Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

In this fork, py-MDNet has been converted from Python 2 to Python 3. 
This means that py-MDNet can be run on Windows for the first time. A dependency of py-MDNet is PyTorch, which is originally designed for Linux and OSX operating systems only. However, the GitHub user 'peterjc123' has made a Windows compatible version [here](https://github.com/peterjc123/pytorch-scripts), but it only runs in Python 3. As a result I converted py-MDNet to Python 3 so that it could be run on Windows.

I ran all .py files through [2to3](https://docs.python.org/2/library/2to3.html#using-2to3) and then fixed all remaining errors. A notable error that I fixed was to add dim=1 paramater to the log_softmax() pytorch function found in modules/model.py line 148

I have further modified the code to provide Accuracy, Robustness and Frames Per Second evaluation metrics similar to those used in the VOT Challenge. To include the Robustness metric, I edited the code so that MDNet will now reinitialize when the overlap between the ground truth bounding box and the estimated bounding box drops to 0. Nskip = 5 frames are skipped after failure. I implemented Accuracy and Robustness according to [this paper.](https://arxiv.org/pdf/1503.01313.pdf)

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
 
## Requirements
- Python 3
- [PyTorch](http://pytorch.org/) and its dependencies 
– Conda
– NumPy
– Numba
– pyCUDA

I recommend using Anaconda and the Anaconda Prompt. 

to install pyCUDA I used:
```bash
pip install pycuda-2017.1.1+cuda9185-cp36-cp36m-win_amd64.whl 
```
[From this site]( https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_PyCUDA_On_Anaconda_For_Windows?lang=en) helped.

To check if CUDA is working, type "numba -s" into the command prompt. This tells you if cuda is working under "__CUDA Information__"

### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
 - Edit /pretrain/vot-training-sequences.txt to include the sequences you wish to train on
``` bash
 cd pretrain
 python prepro_data.py
 python train_mdnet.py
```

### Tracking
```bash
 cd tracking
 python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```

An example if you have downloaded a VOT dataset into /dataset:
```bash
 cd tracking
 python run_tracker.py -s vot2016/matrix -d 
```


 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python run_tracker.py -s [seq name]```
   - ```python run_tracker.py -j [json path]```
 

