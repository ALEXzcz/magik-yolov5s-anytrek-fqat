

#include <fstream>
#include <iostream>
#include <map>
#include <errno.h>
#include <string.h>
#include <string>
#include <stdlib.h>

#include "ingenic_mnni.h"

using namespace std;
using namespace magik::uranus;

int main(int argc , char** argv) {
    // int ret = 0;
    /* set cpu affinity */
    // cpu_set_t mask;
    // CPU_ZERO(&mask);
    // CPU_SET(0, &mask);
    // if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
    //     fprintf(stderr, "set cpu affinity failed, %s\n", strerror(errno));
    //     return -1;
    // }
    // ingenic_aip_init();
    // ret = magik::uranus::uranus_init();
    // if (0 != ret) {
    //     fprintf(stderr, "uranus init failed.\n");
    //     return ret;
    // }

    /******************load model 1***********************/
	std::string mgk_file_name = argv[1];
    Network** derived_models;
    int model_num = 0;
    if (magik_load_model(derived_models, model_num, mgk_file_name.c_str())) {
        printf("create model failed.\n");
        return -1;
    }
    auto model = derived_models[0];

	/******************load model 2***********************/
	// Network* derived_model;
    // std::string mgk_file_name = argv[1];
    // if (magik_load_model(derived_model, mgk_file_name.c_str())) {
    //     printf("create model failed.\n");
    //     exit(1);
    // }
    // auto model = derived_model;

    auto input_names = model->get_input_names();
    for(auto n: input_names) {
        auto input = model->get_input(n);
        uint8_t *w_ptr = (uint8_t*)(input.vdata<uint8_t>());
        FILE *fp = fopen(std::string("./datamodel/"+n +".input").c_str(), "rb");
        if (fp) {
            fread(w_ptr, 1, (input.get_bytes_size()), fp);
            fclose(fp);
        } else {
            printf("open %s file failed\n", std::string(std::string("./datamodel/"+n +".data")).c_str());
            return -1;
        }
    }

    int run_times = 1;
    for (int i = 0; i < run_times; i++)
        model->run();

    auto output_names = model->get_output_names();
    for(auto n: output_names) {
        auto output = model->get_output(n);

        const unsigned char* r_ptr = output.vdata<uint8_t>();
        FILE *fp = fopen(std::string(n+".res").c_str(), "wb");
        if (fp) {
            fwrite(r_ptr, 1, (output.get_bytes_size()), fp);
            fclose(fp);
        } else {
            printf("open %s file failed\n", std::string(n+".res").c_str());
            return -1;
        }
    }
    magik_destroy_model(model);
    /*****************************************************/
    // ret = magik::uranus::uranus_deinit();
    // if (0 != ret) {
    //     fprintf(stderr, "uranus deinit failed.\n");
    //     return ret;
    // }
    // ingenic_aip_deinit();
    return 0;
}

