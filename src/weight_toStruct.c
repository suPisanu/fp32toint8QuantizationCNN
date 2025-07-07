#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "weight_int8_export.h"

typedef struct
{
    int8_t Kernel[CONV_PE_REAL_0][CONV_PE_REAL_0];
} conv_pe_pod_real;

typedef struct
{
    conv_pe_pod_real in_Channel[CONV_INPUT_CH_0];
} conv_input_ch_0;

typedef struct
{
    conv_input_ch_0 filter_Channel[CONV_FILTER_CH_0];
} conv_filter_ch_0;

typedef struct
{
    conv_pe_pod_real in_Channel[CONV_FILTER_CH_1];
} conv_input_ch_1;

typedef struct
{
    conv_input_ch_1 filter_Channel[CONV_FILTER_CH_1];
} conv_filter_ch_1;

typedef struct
{
    conv_filter_ch_0 kernel_layer_0;
    conv_filter_ch_1 kernel_layer_1;
} CNNModel;

int main()
{
    CNNModel MyModel;

    // FIRST CONVOLUTIONAL LAYER
    for (int i = 0; i < CONV_INPUT_CH_0; i++)
    {
        for (int j = 0; j < CONV_FILTER_CH_0; j++)
        {
            for (int k = 0; k < CONV_PE_REAL_0; k++)
            {
                for (int m = 0; m < CONV_PE_REAL_0; m++)
                {
                    MyModel.kernel_layer_0.filter_Channel[i].in_Channel[j].Kernel[k][m] = kernel_layer_0_weight[i][j][k][m];
                    // printf("%d, \n", conv1_weight[i][j][k][m]);
                    // printf("%d, \n", MyModel.Conv1.out_Channel[i].in_Channel[j].Kernel[k][m]);
                }
            }
        }
    }

    // SECOND CONVOLUTIONAL LAYER
    for (int i = 0; i < CONV_INPUT_CH_1; i++)
    {
        for (int j = 0; j < CONV_FILTER_CH_1; j++)
        {
            for (int k = 0; k < CONV_PE_REAL_1; k++)
            {
                for (int m = 0; m < CONV_PE_REAL_1; m++)
                {
                    MyModel.kernel_layer_1.filter_Channel[i].in_Channel[j].Kernel[k][m] = kernel_layer_1_weight[i][j][k][m];
                    // printf("%d, \n", conv1_weight[i][j][k][m]);
                    // printf("%d, \n", MyModel.Conv1.out_Channel[i].in_Channel[j].Kernel[k][m]);
                }
            }
        }
    }

    // Save Struct as Binary File
    // FILE *f = fopen("model_weight.bin", "wb");
    // fwrite(&MyModel, sizeof(CNNModel), 1, f);
    // fclose(f);

    // FILE *f = fopen("model_weight.bin", "rb");
    // if (f == NULL)
    //{
    //     perror("Failed to open file.");
    //     return 1;
    // }
    // fread(&MyModel, sizeof(CNNModel), 1, f);
    // fclose(f);

    // FIRST CONVOLUTIONAL LAYER OUTPUTS
    // for (int i = 0; i < CONV_INPUT_CH_0; i++)
    //{
    //    for (int j = 0; j < CONV_FILTER_CH_0; j++)
    //    {
    //        for (int k = 0; k < CONV_PE_REAL_0; k++)
    //        {
    //            for (int m = 0; m < CONV_PE_REAL_0; m++)
    //            {
    //                // MyModel.kernel_layer_0.filter_Channel[i].in_Channel[j].Kernel[k][m] = kernel_layer_0_weight[i][j][k][m];
    //                //  printf("%d, \n", conv1_weight[i][j][k][m]);
    //                printf("%d, \n", MyModel.kernel_layer_0.filter_Channel[i].in_Channel[j].Kernel[k][m]);
    //            }
    //        }
    //    }
    //}

    // SECOND CONVOLUTIONAL LAYER OUTPUTS
    for (int i = 0; i < CONV_INPUT_CH_1; i++)
    {
        for (int j = 0; j < CONV_FILTER_CH_1; j++)
        {
            for (int k = 0; k < CONV_PE_REAL_1; k++)
            {
                for (int m = 0; m < CONV_PE_REAL_1; m++)
                {
                    // MyModel.kernel_layer_1.filter_Channel[i].in_Channel[j].Kernel[k][m] = kernel_layer_1_weight[i][j][k][m];
                    //  printf("%d, \n", conv1_weight[i][j][k][m]);
                    printf("%d, \n", MyModel.kernel_layer_1.filter_Channel[i].in_Channel[j].Kernel[k][m]);
                }
            }
        }
    }
}
