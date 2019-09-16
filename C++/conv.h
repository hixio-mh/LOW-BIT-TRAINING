#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "math.h"
#include "ap_int.h"

#define img_num 16
#define moments (int)4294967296 

// Quantize forward output
#define CONV_FORWARD 0
#define BATCH_FORWARD 0 
#define SCALE_FORWARD 0 
#define FULL_FORWARD 0 

// Quantize backward indiff
#define FULL_BACKWARD 0 
#define SCALE_BACKWARD 0 
#define BATCH_BACKWARD 0 
#define CONV_BACKWARD 0 

// Quantize weight and bias
#define CONV_WEIGHT 0
#define SCALE_WEIGHT 0 
#define SCALE_BIAS 0  
#define FULL_WEIGHT 0 
#define FULL_BIAS 0

// Show debug information
#define TEST_MODE 0
#define PRINT_LOSS 1
#define PRINT_ACCURACY 1
#define SHOW_PAIRS 0 

// Show forward results
#define SHOW_CONV_IN 0
#define SHOW_BATCH_IN 0
#define SHOW_SCALE_IN 0
#define SHOW_RELU_IN 0
#define SHOW_POOL_IN 0
#define SHOW_FULL_IN 0
#define SHOW_SOFTMAX_IN 0
#define SHOW_SOFTMAX_OUT 0

// Show backward results (diff)
#define SHOW_CONV_INDIFF 0
#define SHOW_BATCH_INDIFF 0
#define SHOW_SCALE_INDIFF 0
#define SHOW_RELU_INDIFF 0
#define SHOW_POOL_INDIFF 0
#define SHOW_FULL_INDIFF 0
#define SHOW_SOFTMAX_INDIFF 0

// Show weights and bias
#define SHOW_CONV_WEIGHT 0
#define SHOW_FULL1_WEIGHT 0
#define SHOW_FULL1_BIAS 0
#define SHOW_FULL2_WEIGHT 0
#define SHOW_FULL2_BIAS 0
#define SHOW_SCALE_WEIGHT 0
#define SHOW_SCALE_BIAS 0

#define SHOW_QUANTIZE_RESULT 0

float input_qt(float input, int s_in, int *larger, int *smaller, int *all, bool stochastic);

void memset_float(float *to, int size);

void quantize_backward(float *to, int size, int *step, bool if_stochastic); 

void quantize_forward(float *from, float *to, int size, int *step, bool if_stochastic);

void training(float *data, float *wg, float *fp, int *setting, int *sp, int rate, int epoch, int decay, float *temp_weight, float *temp_bias, int flag);


void conv(float *input, float *weights, float *tmp, float *output, int *params, int *sparams);

void conv_back(float *outdiff, float *weights, float *indiff, float *tmp, int *params, int *sparams);

void weight_back(float *outdiff, float *weights, float *input, float *moment, int *params, int *sparams, int rate);



void fc(float *input, float *weights, float *bias,
    float *output, int *params, int *sparams);

void back_fc(float *outdiff, float *weights, float *tmp,
    float *indiff, int *params, int *sparams);

void fc_weight(float *outdiff, float *weights,  float *input, float *moment, int *params, int *sparams, int rate);

void bias_back_fc(float *outdiff, float *bias, float *moment, int *params, int *sparams, int rate);


void pool(float *input, float *pos,
    float *output, int *params, int *sparams);

void back_pool(float *indiff, float *pos,
    float *outdiff0, float *outdiff1, int *params, int *sparams);


void relu(float *input,
    float *output, int *params, int *sparams);

void relu_back(float *input,
    float *outdiff0, float *outdiff1, float *indiff, int *params, int *sparams);


void eltwise(float *input0, float *input1, float *output, int *params);

void eltwise_back(float *outdiff, float *indiff, int *params);


void batch(float *input, float *temp,
  float *output, int *params, int *sparams);

void batch_back(float *output, float *indiff, float *temp,
    float *outdiff,  int *params, int *sparams);


void scale(float *input, float *weights, float *bias,
    float *output, int *params, int *sparams);

void scale_back(float *indiff, float *weights, float *outdiff, int *params, int *sparams); 

void weight_scale(float *outdiff, float *weights,  float *input, float *moment, int *params, int *sparams, int rate);

void bias_scale(float *bias, float *outdiff, float *moment, int *params, int *sparams, int rate);


void softmax(float *input, float *output, int *params, int *sparams, float *diff, float *tmp);

#endif
