#ifndef _FACENET_H_
#define _FACENET_H_

void* init_facenet_network_file(const char *mpath);

int set_facenet_input(void *qcontext, cv::Mat orig_img, int input_width, int input_high, int input_channel);

int run_facenet_network(void *qcontext, float** result);

int destroy_facenet_network(void *qcontext);

#endif //FACENET_H
