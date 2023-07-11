# Compile

```shell
$ sudo apt install libopencv-dev python3-opencv cmake
$ mkdir build && cd build
$ cmake ..
$ make
```

# Run

```shell
$ cd build
$ sudo ./yolov8n -p ../data/horses.jpg  -m ../data/yolov8n_int8.adla
```
