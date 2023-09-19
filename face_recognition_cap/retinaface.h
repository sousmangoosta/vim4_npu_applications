#ifndef _RETINAFACE_H_
#define _RETINAFACE_H_

void* init_retinaface_network_file(const char *mpath);

int set_retinaface_input(void *qcontext, cv::Mat orig_img, int input_width, int input_high, int input_channel);

int run_retinaface_network(void *qcontext, face_detect_out_t &retinaface_detect_out);

int destroy_retinaface_network(void *qcontext);

#endif //RETINAFACE_H
