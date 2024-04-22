#include <cmath>
#include "nn_sdk.h"
#include "postprocess_util.h"
#include <vector>

float iou(const dBox boxA, const dBox boxB) {
    const float eps = 1e-6;
    float iou = 0.f;
    float areaA = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin);
    float areaB = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin);
    float x1 = std::max(boxA.xmin, boxB.xmin);
    float y1 = std::max(boxA.ymin, boxB.ymin);
    float x2 = std::min(boxA.xmax, boxB.xmax);
    float y2 = std::min(boxA.ymax, boxB.ymax);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    iou = inter / (areaA + areaB - inter + eps);
    return iou;
}

int compare(const void *pa, const void *pb) {
    dBox a = *(dBox *) pa;
    dBox b = *(dBox *) pb;
    return a.score + a.objectClass < b.score + b.objectClass;
}


void nms(dBox *tBox, float iou_threshold, int rows) {
    qsort(tBox, rows, sizeof(dBox), compare);
    for (int i = 0; i < rows; i++) {
        if (tBox[i].score == 0.f) {
            continue;
        }
        for (int j = i + 1; j < rows; ++j) {
            if (tBox[i].objectClass != tBox[j].objectClass) {
                break;
            }
            if (iou(tBox[i], tBox[j]) > iou_threshold) {
                tBox[j].score = 0.f;
            }
        }
    }
}

void postprocess_ssd(nn_output *pout, unsigned int *detNum, dBox *pBox) {
    float *ssd_buffer[3] = {nullptr};
    ssd_buffer[0] = (float *) pout->out[0].buf;
    ssd_buffer[1] = (float *) pout->out[1].buf;
    ssd_buffer[2] = (float *) pout->out[2].buf;
    dBox tBox[MAX_DETECT_NUM];

    struct Anchor {
        double y;
        double x;
        double h;
        double w;
    } anchor{};
    struct BoxEncodings {
        double y;
        double x;
        double h;
        double w;
    } boxEncodings{};
    struct Box {
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    } box{};
    int num_detections = 0;
    for (int i = 0; i < 2034; i++) {
        double y_scale = 10.0;
        double x_scale = 10.0;
        double h_scale = 5.0;
        double w_scale = 5.0;
        boxEncodings.y = (double) ssd_buffer[1][i * 4 + 0];
        boxEncodings.x = (double) ssd_buffer[1][i * 4 + 1];
        boxEncodings.h = (double) ssd_buffer[1][i * 4 + 2];
        boxEncodings.w = (double) ssd_buffer[1][i * 4 + 3];
        anchor.y = (double) ssd_buffer[2][i * 4 + 0];
        anchor.x = (double) ssd_buffer[2][i * 4 + 1];
        anchor.h = (double) ssd_buffer[2][i * 4 + 2];
        anchor.w = (double) ssd_buffer[2][i * 4 + 3];
        auto ycenter = (float) ((double) boxEncodings.y / (double) y_scale * (double) anchor.h + (double) anchor.y);
        auto xcenter = (float) ((double) boxEncodings.x / (double) x_scale * (double) anchor.w + (double) anchor.x);
        auto half_h = (float) (0.5 * (exp((double) boxEncodings.h / (double) h_scale)) * (double) anchor.h);
        auto half_w = (float) (0.5 * (exp((double) boxEncodings.w / (double) w_scale)) * (double) anchor.w);
        box.ymin = ycenter - half_h;
        box.xmin = xcenter - half_w;
        box.ymax = ycenter + half_h;
        box.xmax = xcenter + half_w;
        for (int j = 0; j < 91; j++) {
            float prob = ssd_buffer[0][i * 91 + j];
            if (prob > .6) {
                tBox[num_detections].ymin = box.ymin;
                tBox[num_detections].xmin = box.xmin;
                tBox[num_detections].ymax = box.ymax;
                tBox[num_detections].xmax = box.xmax;
                tBox[num_detections].score = prob;
                tBox[num_detections].objectClass = (float) j - 1;
                num_detections++;
            }
        }
    }

    nms(tBox, 0.6000000238418579, num_detections);
    int nms_detections = 0;
    for (int i = 0; i < num_detections; i++) {
        if (tBox[i].score > 0.f) {
            pBox[nms_detections] = tBox[i];
            nms_detections++;
        }
    }
    *detNum = nms_detections;
}
