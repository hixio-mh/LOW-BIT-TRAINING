#include "conv.h"

void eltwise(float *input0, float *input1, float *output, int *params, int *sparams)
{
	float in_buf0[PARA][img_num];
	float in_buf1[PARA][img_num];
	float out_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	for(int o = 0; o<iteration_out; o++)
		for(int y = 0; y < y_dim; y++)
			for(int x = 0; x < x_dim; x++)
			{
				#pragma omp parallel for
				for(int p = 0; p<PARA; p++)
				{
					int in_trans_size = img_num;
					int in_off = ((y*x_dim+x)*inChannel+o*PARA + p)*img_num;
					memcpy(in_buf0[p], input0 + in_off, in_trans_size*4);
					memcpy(in_buf1[p], input1 + in_off, in_trans_size*4);

					for(int im = 0; im<img_num;im++)
						out_buf[p][im] = in_buf0[p][im] + in_buf1[p][im];

					int out_trans_size = img_num;
					int out_off = ((y*x_dim+x)*inChannel+o*PARA+p)*img_num;
					memcpy(output + out_off, out_buf[p], in_trans_size*4);
				}
			}
}

void eltwise_back(float *outdiff, float *indiff, int *params, int *sparams)
{
	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

 	for(int o = 0; o<iteration_out; o++)
		for(int y =0; y < y_dim; y++)
	  		for(int x = 0; x < x_dim;x++)
	 		{
	    		int out_trans_size = PARA*img_num;
	    		int out_off = ((y*x_dim+x)*inChannel + o*PARA)*img_num;
	    		memcpy(indiff + out_off, outdiff + out_off, out_trans_size*4);
     		}  
}
