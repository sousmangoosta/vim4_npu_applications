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
$ sudo ./mobilenetv2 ../data/mobilenetv2_int8.adla ../data/goldfish_224x224.jpg
```
