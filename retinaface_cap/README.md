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
$ sudo ./retinaface_cap -m ../data/retinaface_int8.adla -d 0 -w 1920 -h 1080
```

**x**: the number for you camera device. such as `/dev/video0`, `x` is `0`.