#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "nn_sdk.h"
#include "postprocess_util.h"
#include "../nn_sdk/include/nn_sdk.h"

extern "C" // let this function callable by "C" like python
void *init_network_file(const char *mpath) {
    void *qcontext;

    aml_config config;
    memset(&config, 0, sizeof(aml_config));

    config.nbgType = NN_ADLA_FILE;
    config.path = mpath;
    config.modelType = ADLA_LOADABLE;
    config.typeSize = sizeof(aml_config);

    qcontext = aml_module_create(&config);
    if (qcontext == nullptr) {
        printf("amlnn_init is fail\n");
        return nullptr;
    }

    if (config.nbgType == NN_ADLA_MEMORY && config.pdata != nullptr) {
        free((void *) config.pdata);
    }

    return qcontext;
}

extern "C" // let this function callable by "C" like python
int set_input(void *qcontext, unsigned char *img_array, int img_size) {
    nn_input inData;
    int ret;

    inData.input_type = RGB24_RAW_DATA;
    inData.input = img_array;
    inData.input_index = 0;
    inData.size = img_size;
    ret = aml_module_input_set(qcontext, &inData);

    return ret;
}

extern "C" // let this function callable by "C" like python
int run_network(void *qcontext, unsigned int *detNum, dBox *pBox) {
    int ret = 0;
    nn_output *outdata;
    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));

    outconfig.format = AML_OUTDATA_RAW;
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.order = AML_OUTPUT_ORDER_NHWC;

    outdata = (nn_output *) aml_module_output_get(qcontext, outconfig);
    if (outdata == nullptr) {
        printf("aml_module_output_get error\n");
        return -1;
    }

    postprocess_ssd(outdata, detNum, pBox);

    return ret;
}

extern "C" // let this function callable by "C" like python
int destroy_network(void *qcontext) {

    int ret = aml_module_destroy(qcontext);
    return ret;
}
