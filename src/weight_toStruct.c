#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "weight_int8_export.h"

typedef struct
{
    int8_t Kernel[CONV1_kernel_Width][CONV1_kernel_Height];
} ConvKernel;

typedef struct
{
    ConvKernel in_Channel[CONV1_in_Channel];
} Conv1Depth;

typedef struct
{
    Conv1Depth out_Channel[CONV1_out_Channel];
} Conv1Filters;

typedef struct
{
    ConvKernel in_Channel[CONV2_in_Channel];
} Conv2Depth;

typedef struct
{
    Conv2Depth out_Channel[CONV2_out_Channel];
} Conv2Filters;

typedef struct
{
    Conv1Filters Conv1;
    Conv2Filters Conv2;
} CNNModel;

int main()
{
    CNNModel MyModel;

    // FIRST CONVOLUTIONAL LAYER
    for (int i = 0; i < CONV1_out_Channel; i++)
    {
        for (int j = 0; j < CONV1_in_Channel; j++)
        {
            for (int k = 0; k < CONV1_kernel_Width; k++)
            {
                for (int m = 0; m < CONV1_kernel_Height; m++)
                {
                    MyModel.Conv1.out_Channel[i].in_Channel[j].Kernel[k][m] = conv1_weight[i][j][k][m];
                    // printf("%d, \n", conv1_weight[i][j][k][m]);
                    // printf("%d, \n", MyModel.Conv1.out_Channel[i].in_Channel[j].Kernel[k][m]);
                }
            }
        }
    
    // SECOND CONVOLUTIONAL LAYER
    for (int i = 0; i < CONV2_out_Channel; i++)
    {
        for (int j = 0; j < CONV2_in_Channel; j++)
        {
            for (int k = 0; k < CONV2_kernel_Width; k++)
            {
                for (int m = 0; m < CONV2_kernel_Height; m++)
                {
                    MyModel.Conv2.out_Channel[i].in_Channel[j].Kernel[k][m] = conv2_weight[i][j][k][m];
                    // printf("%d, \n", conv2_weight[i][j][k][m]);
                    // printf("%d, \n", MyModel.Conv2.out_Channel[i].in_Channel[j].Kernel[k][m]);
                }
            }
        }
    }

    //Save Struct as Binary File
    //FILE *f = fopen("model_weight.bin", "wb");
    //fwrite(&MyModel, sizeof(CNNModel), 1, f);
    //fclose(f);
    
    //FILE *f = fopen("model_weight.bin", "rb");
    //if (f == NULL)
    //{
    //    perror("Failed to open file.");
    //    return 1;
    //}
    //fread(&MyModel, sizeof(CNNModel), 1, f);
    //fclose(f);
}