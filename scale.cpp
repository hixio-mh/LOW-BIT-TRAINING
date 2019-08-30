#include "conv.h"

#define PARA 10

void scale(float *input, float *weights, float *bias, float *output, int *params, int *sparams){

	float weight_buf[PARA];
	float bias_buf[PARA];
	float in_buf[PARA][img_num];
	float out_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

   	for(int i = 0; i<iteration_in; i++)
    {
		int wg_trans_size= PARA;
		int wg_offset = i*PARA;
		memcpy(weight_buf, weights + wg_offset, wg_trans_size*4);
		memcpy(bias_buf, bias + wg_offset, wg_trans_size*4);

    	#pragma omp parallel for
    	for(int p = 0; p<PARA; p++)
		{
			for(int y =	0; y < y_dim; y++)
				for(int x = 0; x < x_dim; x++)
				{
					int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
					memcpy(in_buf[p],  input + in_offset, img_num*4);

					for(int im = 0; im<img_num; im++)
					{
           				out_buf[p][im] = in_buf[p][im]*weight_buf[p] + bias_buf[p];  

        			}

        			for(int im = 0; im<img_num; im++)
						output[in_offset + im] = out_buf[p][im];
    			}  
    	}
	}
}

void scale_back(float *indiff, float *weights,float *outdiff, int *params, int *sparams){

	float weight_buf[PARA];
	float bias_buf[PARA];
	float in_buf[PARA][img_num];
	float out_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

  	for(int i = 0; i<iteration_in; i++)
	{
		int wg_trans_size = PARA;
		int wg_offset = i*PARA;
		memcpy(weight_buf, weights + wg_offset, wg_trans_size*4);

		#pragma omp parallel for
		for(int p = 0; p<PARA; p++)
			for(int im = 0; im<img_num; im++)
				for(int y = 0; y < y_dim; y++){
					for(int x = 0; x < x_dim; x++){
			
						int in_trans_size = img_num;
						int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num+im;  
						out_buf[p][im] = outdiff[in_offset]*weight_buf[p];   
						indiff[in_offset] = out_buf[p][im];
					}   
				}
	}	
}

void weight_scale(float *input, float *weights, float *outdiff, float *moment, int *params, int *sparams, int rate){

	float weight_buf[PARA];
	float in_buf[PARA][img_num];
	float moment_buf[PARA];
	float out_buf[PARA][img_num];
	float sum[PARA];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	for(int i = 0; i<iteration_in; i++)
	{
		int wg_trans_size= PARA;
		int wg_offset = i*PARA;
		memcpy(weight_buf, weights + wg_offset, wg_trans_size*4);
		memcpy(moment_buf, moment + wg_offset, wg_trans_size*4);

  		#pragma omp parallel for
	 	for(int p = 0; p<PARA; p++)
	 	{
    		sum[p] = 0;
			for(int y = 0; y < y_dim; y++)
		  		for(int x = 0; x < x_dim; x++)
		 		{		
					int in_trans_size = img_num;
					int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
					memcpy(in_buf[p], input + in_offset, in_trans_size*4);
					memcpy(out_buf[p], outdiff + in_offset, in_trans_size*4);

					for(int im=0;im<img_num;im++)
				  		sum[p] += out_buf[p][im]*in_buf[p][im]; 

				}

    		moment_buf[p] = 0.9*moment_buf[p] + sum[p]/rate; 
    		weights[i*PARA+p] = weight_buf[p] - moment_buf[p];
    
	 	}

		memcpy(moment + wg_offset, moment_buf, wg_trans_size*4);
 	}
}

void bias_scale(float *bias, float *outdiff, float * moment, int *params, int *sparams, int rate){

 	float bias_buf[PARA];
 	float moment_buf[PARA];
 	float in_buf[PARA][img_num];
 	float out_buf[PARA][img_num];
 	float sum[PARA];

 	int inChannel = params[0];
 	int outChannel = params[1];
 	int inCh_once = params[2];
 	int outCh_once = params[3];
 	int iteration_in = inChannel/PARA;
 	int iteration_out = outChannel/PARA;
 	int x_dim = params[4];
 	int y_dim = params[5];
 	int stride = params[6];
 	int pad = params[7];
 	int ksize = params[8];
	int maxpool = params[9];

  	for(int i = 0; i<iteration_in; i++)
 	{
		int bias_trans_size= PARA;
		int bias_offset = i*PARA;
		memcpy(bias_buf, bias + bias_offset, bias_trans_size*4);
		memcpy(moment_buf , moment + bias_offset, bias_trans_size*4);
 	
	 	#pragma omp parallel for
 	 	for(int p = 0; p<PARA; p++)
 	 	{
    		sum[p] = 0;
			for(int y = 0; y < y_dim; y++)
				for(int x = 0; x < x_dim; x++)
				{
 					memcpy(out_buf[p] , outdiff + ((y*x_dim+x)*inChannel+i*PARA+p)*img_num, img_num*4);
            		for(int im = 0 ;im<img_num; im++)
            			sum[p] += out_buf[p][im];
 		 		}

			moment_buf[p] = 0.9*moment_buf[p] + sum[p]/rate;
			bias[i*PARA+p] = bias_buf[p] - moment_buf[p];
 		}

 		memcpy(moment + bias_offset, moment_buf, bias_trans_size*4);
  	}
}




