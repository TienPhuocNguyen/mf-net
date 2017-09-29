# ...



This repository provides demo code for:
- Training a triplet model to learn motion feature.
- Computing segmentation masks for any image sequence with an extracted MF-Net.

## Repository Structure

The repository is structured as follows:
- `models` contains the Triplet model to train our network
- `net` contains our trained model
- `utils` contain lua and CUDA code for our Change Detection algorithm. This needs to be compiled before execution.

### Dependencies

This code requires:
- [torch7](https://github.com/torch/torch7)
- [opencv >= 2.4](http://opencv.org/)(for training)
- [png++](http://www.nongnu.org/pngpp/)  (for training)
- [cudnn](https://developer.nvidia.com/cudnn)
- display package for torch7

## Compute the Segmentation Mask
Download this repository:
 ~~~
$git clone https://github.com/TienPhuocNguyen/mf-net
 ~~~
First, we have to compile the shared libraries:
 ~~~
 $cd mf-net/utils
 $make
 ~~~
If errors occur, you should check the computing architecture of your GPU and modify the `Makefile`. Here, we use `sm_61` for the Titan X Pascal.
If success, the command will produce a file `libcutils.so`.

To display results on your screen, install the `display` package:
 ~~~
 $luarocks install display
 ~~~
Launch the server:
 ~~~
 $th -ldisplay.start 8000 0.0.0.0
 ~~~
Then open `0.0.0.0:8000` on your browser to open the remote desktop.

To execute the program, run the command:
 ~~~
 $th run.lua
 ~~~
The file `run.lua` also provides some arguments to specify the directories of image sequence and trained model. For examples:
 ~~~
 $th run.lua -n net/trained.t7 -s datasets/CDNet2014/dataset/dynamicBackground/fall/input
 ~~~

## TODO
We will release the training code for this paper soon.





