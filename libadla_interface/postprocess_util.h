#ifndef POSTPROCESS_UTIL_H_
#define POSTPROCESS_UTIL_H_

typedef struct lib_box {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
    float score;
    float objectClass;
} dBox;

void postprocess_ssd(nn_output *pout, unsigned int *detNum, dBox *pBox);

#endif //POSTPROCESS_UTIL_H_
