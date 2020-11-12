# mobilenetv1-ssd-tensorrt

* This project is based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd). The project has been tested on TensorRT 7.0 CUDA 10.2 CUDNN 7.6.5, and costs about 1ms(1000fps) to inference an image on GeForce GTX 1660 Ti.

* The project also has been tested on TensorRT 7.1.0(Developer Preview) CUDA 10.2 CUDNN 8.0.0(Developer Preview), and costs about 10-12ms(83-100fps) to inference an image on TX2 (by using the MAX-N mode and jetson_clocks).

* Another project ["yolov4-tiny-tensorrt"](https://github.com/tjuskyzhang/yolov4-tiny-tensorrt).

## Excute:

(1) Generate mobilenet-v1-ssd.wts from pytorch implementation

```
  git clone https://github.com/tjuskyzhang/mobilenetv1-ssd-tensorrt.git
  
  git clone https://github.com/qfgaohao/pytorch-ssd.git
  
  cd pytorch-ssd
  
  wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
  
  wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
``` 
// 权重下载链接：https://pan.baidu.com/s/1Nagw-qP_PdTG4u_a9Dml-Q 提取码：yg27  
```
  cp ../mobilenetv1-ssd-tensorrt/gen_wts.py .

  python gen_wts.py
```
// A file named 'mobilenet-v1-ssd.wts' will be generated.

```
  cp models/mobilenet-v1-ssd.wts ../mobilenetv1-ssd-tensorrt
```

(2) Build and run

```
  cd mobilenetv1-ssd-tensorrt

  mkdir build

  cd build

  cmake ..

  make
```
// Serialize the model and generate ssd_mobilenet.engine
```
  ./mobilenet-ssd-tensorrt -s
```

// Deserialize and generate the detection results _dog.jpg and so on.

```
  ./mobilenet-ssd-tensorrt -d ../samples
```
