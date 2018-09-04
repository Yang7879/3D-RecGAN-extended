# Dense 3D Object Reconstruction from a Single Depth View
Bo Yang, Stefano Rosa, Andrew Markham, Niki Trigoni, Hongkai Wen. [TPAMI](http://dx.doi.org/10.1109/TPAMI.2018.2868195), 2018.

## (1) Architecture
![Arch_Image](https://github.com/Yang7879/3D-RecGAN-extended/blob/master/3D-RecGAN%2B%2B_arch.png)
## (2) Sample Results
![Teaser_Image](https://github.com/Yang7879/3D-RecGAN-extended/blob/master/3D-RecGAN%2B%2B_sample.png)

## (3) Data
#### Part 1: {ShapeNetCore.v2: bench, chair, couch, table}, 20G
[https://drive.google.com/open?id=1rmOggF0ivB42KozMX3sQGD1CkZNOGCmM](https://drive.google.com/open?id=1rmOggF0ivB42KozMX3sQGD1CkZNOGCmM)
#### Part 2: {ShapeNetCore.v2: airplane, car, monitor, faucet, guitar, gun}, 9.3G
[https://drive.google.com/open?id=1zLQd68O73ZiwZ8S8qsLwwGYDcC5PiEdG](https://drive.google.com/open?id=1zLQd68O73ZiwZ8S8qsLwwGYDcC5PiEdG)
#### Real Dataset: {Kinect: bench, chair, couch, table}
[https://drive.google.com/open?id=1wTE721q0r66Z6yyN68O1Tz4Bg5-aYnq3](https://drive.google.com/open?id=1wTE721q0r66Z6yyN68O1Tz4Bg5-aYnq3)

## (4) Released Model
#### Trained on {bench, chair, couch, table}, 2G
[https://drive.google.com/open?id=1IzwZLgRhzd6GVofzdjBZTblxMPH7NuxP](https://drive.google.com/open?id=1IzwZLgRhzd6GVofzdjBZTblxMPH7NuxP)

## (5) Requirements
python 2.7.6

tensorflow 1.2.0

numpy 1.13.3

scipy 0.19.0

matplotlib 2.0.2

skimage 0.13.0

## (6) Run
#### Training
python main_3D-RecGAN++.py

#### Test Demo (Download released model first)
python demo_3D-RecGAN++.py

## (7) Citation
If you use the paper, code or data for your research, please cite:
```
@inProceedings{Yang18,
  title={Dense 3D Object Reconstruction from a Single Depth View},
  author = {Bo Yang
  and Stefano Rosa
  and Andrew Markham
  and Niki Trigoni
  and Hongkai Wen},
  booktitle={TPAMI},
  year={2018}
}
```
