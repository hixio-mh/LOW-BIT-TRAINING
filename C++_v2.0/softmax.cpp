#include "conv.h"

void softmax(float *input, float *output, int *params, int *sparams, float *diff, float *tmp)
{
	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/inCh_once;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	float sum[img_num];
	float chn[inCh_once][img_num];
	float tm[img_num];
	float  in_buf[inCh_once][img_num];
	float in_diffbuf[inCh_once][img_num];
	float resbuf[inCh_once][img_num];
	float  diffbuf[inCh_once][img_num];
	float  out_buf[inCh_once][img_num];
    float  factor[inCh_once];
    float  mean[inCh_once];
    float  var[inCh_once];
    float  out_tmp[inCh_once][img_num];
    
    int x_out = x_dim; 
    int y_out = x_out;
   
	for(int i = 0; i<iteration_in; i++)
	{
		for(int p = 0; p<inCh_once; p++)
		{
		    int in_trans_size= img_num;
		    int in_offset = (i*inCh_once+p)*img_num;
		    memcpy(in_buf[p], input + in_offset, in_trans_size*4);
		    
		    float max = in_buf[p][0];
		    for(int im =1; im<img_num; im++)
         	{
     			if(in_buf[p][im] > max);
     				max = in_buf[p][im];
     		}
     		
            #pragma omp parallel for
            for(int im = 0; im<img_num; im++)
         	{
				tm[im] = exp(in_buf[p][im]);
				if(i == 0 && p == 0)
					sum[im] = tm[im];
				else
					sum[im] += tm[im];
     		}
    		memcpy(tmp+in_offset, tm, 4*in_trans_size);
		}
	}

	for(int i = 0; i<iteration_in; i++)
	{
		#pragma omp parallel for
		for(int p = 0; p<inCh_once; p++)
		{	
			int in_trans_size= img_num;
			int in_offset = (i*inCh_once+p)*img_num;
			memcpy(chn[p], tmp + in_offset, 4*in_trans_size);
			memcpy(out_buf[p], output + in_offset, in_trans_size*4);

			for(int im = 0; im<img_num; im++)
			{
				in_diffbuf[p][im] = chn[p][im]/sum[im];
				resbuf[p][im] = chn[p][im]/sum[im];
				in_diffbuf[p][im] -= out_buf[p][im];
				if(TEST_MODE)
					diff[(i*inCh_once+p)*img_num +im] = in_diffbuf[p][im];
				else
					diff[(i*inCh_once+p)*img_num +im] = in_diffbuf[p][im]/img_num;
			}
			
		}
        memcpy(tmp+ i*inCh_once*img_num , resbuf, inCh_once*img_num*4);
	}
}


