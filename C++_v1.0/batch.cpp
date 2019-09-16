#include "conv.h"

#define PARA 10

void batch(float *input, float *temp, float *output, int *params, int *sparams){

	float weight_buf[PARA][1024];
	float in_buf[PARA][img_num];
	float bias_buf[PARA][2048/PARA];
	float factor[PARA];
	float sum[PARA];
	float mean[PARA];
	float var[PARA];
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

	        		for(int im = 0; im<img_num; im++)
	        		{
	           			sum[p] += in_buf[p][im];
	        		}
    	 		}
    		mean[p] = sum[p]/x_dim/y_dim/img_num;
			//printf("%d %f\n", p, mean[p]);
       		sum[p] = 0;
			for(int y = 0; y < y_dim; y++)
				for(int x = 0; x < x_dim; x++)
				{
					int in_trans_size = img_num;
					int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
					memcpy(in_buf[p], input + in_offset, in_trans_size*4);

					for(int im = 0; im<img_num; im++)
					{
						float tmp = in_buf[p][im]-mean[p];  
						sum[p] += tmp*tmp;
					}
				}	
			var[p] = sum[p]/x_dim/y_dim/img_num;
  			//printf("%d %f\n", p, var[p]);
      		float e = 1e-5;
			factor[p] = sqrt(var[p] + e);

      		for(int y =	0; y < y_dim; y++)
		    	for(int x = 0; x < x_dim; x++)
		     	{
				 	int in_trans_size = img_num;
				 	int in_offset = ((y*x_dim+x)*inChannel+i*PARA+p)*img_num;
				 	memcpy(in_buf[p], input + in_offset, in_trans_size*4);
					for(int im = 0; im<img_num; im++)
				  	{
           				float res_fp = (in_buf[p][im]-mean[p])/factor[p];
						out_buf[p][im] = res_fp;
					
					}
					memcpy(output + in_offset, out_buf[p], in_trans_size*4);
				}
		} 
    	memcpy(temp + i * PARA, factor, PARA*4);
 	}
}



void batch_back(float *output, float *indiff, float *temp, float *outdiff, int *params, int *sparams) {

    float out_buf[PARA][img_num];
    float dif_buf[PARA][img_num];
    float sum0[PARA];
    float sum1[PARA];
    float mean0[PARA];
    float mean1[PARA];
    float factor[PARA];
    float outdiff_buf[PARA][img_num];

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
      	for(int p = 0; p<PARA; p++) 
        {
			sum0[p] = 0;
         	sum1[p] = 0;
		}
    	
     	for(int y = 0; y < y_dim; y++)
    	  	for(int x = 0; x < x_dim; x++)
    	 	{
				int in_trans_size = PARA;
				int in_offset = ((y*x_dim+x)*inChannel+i*PARA);
				memcpy(out_buf, output + in_offset*img_num, in_trans_size*img_num*4);
		    	memcpy(dif_buf, outdiff + in_offset*img_num, in_trans_size*img_num*4);

				#pragma omp parallel for
		    	for(int p = 0; p<PARA; p++)
		    	{ 
           			for(int im = 0; im < img_num; im++)
            		{
						sum0[p]+= dif_buf[p][im];
             			sum1[p]+= dif_buf[p][im]*out_buf[p][im];
					}
	      		}
    	  	}

      		#pragma omp parallel for
    	for(int p = 0; p<PARA; p++)
    	{
    	    mean0[p] = sum0[p]/x_dim/y_dim/img_num;
          	mean1[p] = sum1[p]/x_dim/y_dim/img_num;
    	}
    	memcpy(factor, temp + i*PARA, PARA*4);

     	for(int y = 0; y < y_dim; y++)
     	  	for(int x = 0; x < x_dim; x++)
     	 	{
     	    	int in_trans_size = PARA;
     		    int in_offset = (y*x_dim+x)*inChannel+i*PARA;
     		    memcpy(out_buf, output + in_offset*img_num, in_trans_size*img_num*4);
     		    memcpy(dif_buf, outdiff + in_offset*img_num, in_trans_size*img_num*4);

            	#pragma omp parallel for
     		    for(int p = 0; p<PARA; p++)
     		    {
            		for(int im = 0; im<img_num; im++){
            			float tm1 = (dif_buf[p][im] - mean0[p] - out_buf[p][im]*mean1[p]) / factor[p];
            			outdiff_buf[p][im] = tm1; 
					}   
 	          	}

     			memcpy(indiff+in_offset*img_num, outdiff_buf, in_trans_size*img_num*4);
     	  	}
    }
}

