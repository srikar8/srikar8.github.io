---
title: YOLOv3 from Darknet to an IR format 
description: YOLOv3 from Darknet to an IR format 
pubDate: Jul 23 2019
heroImage: /blog-placeholder-3.jpg
---

## Method 1

## Clone the Repository

```c++
$ git clone https://github.com/mystic123/tensorflow-yolo-v3.git
```

### YOLOv3 Darknet to YOLOv3 TensorFlow Model

```c++
$ cd tensorflow-yolo-v3/

$ python3 convert_weights_pb.py --class_names /home/srikar/Documents/srikar/Convert_Darknet_to_IR/Darknet_model/coco.names --data_format NHWC --weights_file /home/srikar/Documents/srikar/Convert_Darknet_to_IR/Darknet_model/yolo-obj_3000.weights
```

### Convert YOLOv3 TensorFlow Model to the IR

```c++
$ cd /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/extensions/front/tf
```

#### Add below lines to yolo_v3.json file.

#### **changed the classes according to the trained model.

```json
[
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 2,
      "coords": 4,
      "num": 9,
      "mask": [0,1,2,3,4,5,6,7,8],
      "jitter":0.3,
      "ignore_thresh":0.7,
      "truth_thresh":1,
      "random":1,
      "anchors":[10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326],
      "entry_points": ["detector/yolo-v3/Reshape", "detector/yolo-v3/Reshape_4", "detector/yolo-v3/Reshape_8"]
    }
  }
]
```

***save the yolo_v3.json file***

----------------------------------------

```c++
$ cd /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer

 $ sudo python3 ./mo_tf.py --input_model /home/srikar/Documents/srikar/object_detection/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --input_shape [1,416,416,3] --data_type FP32

```



> [ SUCCESS ] Generated IR model.
> [ SUCCESS ] XML file: /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/./frozen_darknet_yolov3_model.xml
> [ SUCCESS ] BIN file: /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/./frozen_darknet_yolov3_model.bin
> [ SUCCESS ] Total execution time: 43.66 seconds.



## Method 2

Using Docker

docker pull ubuntu:latest

docker pull srikar8/openvino:latest

### YOLOv3 Darknet to YOLOv3 TensorFlow Model


```
docker run --rm -v /home/srikar/Documents/srikar/Convert_Darknet_to_IR/Darknet_model:/app ubuntu:latest /bin/bash -c 'apt-get update; apt-get install -y git; git clone https://github.com/mystic123/tensorflow-yolo-v3.git; cd tensorflow-yolo-v3/; apt install -y python3-pip; pip3 install  numpy; pip3 install tensorflow==1.12.0; pip3 install  pillow; python3 convert_weights_pb.py --class_names /app/coco.names --data_format NHWC --weights_file /app/yolo-obj_3000.weights;cp frozen_darknet_yolov3_model.pb /app'

```

### Convert YOLOv3 TensorFlow Model to the IR

```
docker run --rm -v /home/srikar/Documents/srikar/Convert_Darknet_to_IR/Darknet_model:/app openvino:latest /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh; cd app; sudo python3 /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/mo_tf.py --input_model /app/frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /app/yolo_v3.json --input_shape [1,416,416,3] --data_type FP32'
```



-----------





### The End