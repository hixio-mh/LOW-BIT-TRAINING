#include "conv.h"

void pool(float *input, float *pos, float *output, int *params, int *sparams)
{
	float in_buf[64][PARA*img_num];

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

	int x_out = ((x_dim - ksize) / stride) + 1;
    int y_out = x_out;

	for(int o = 0; o < iteration_out; o++)
		for(int y = 0; y < y_out; y++)
	    	for(int x = 0; x < x_out; x++)
	    	{
	            for(int p = 0; p<ksize; p++)
	        		for(int q = 0; q<ksize; q++)
	        		{
	        			int x_in = x*stride+q;
						int y_in = y*stride+p;
						int inoff = p*ksize+q;
			    		int insize = PARA*img_num;
						if((stride + q < ksize) && x != 0)
						{
							memcpy(in_buf[inoff], in_buf[p*ksize+stride+q], insize*4);
						}
						else
						{
							int in_off = ((y_in*x_dim+x_in)*inChannel+o*PARA)*img_num;
							memcpy(in_buf[inoff], input + in_off, insize*4);                                           
						}
					}
                  
				#pragma omp parallel for
	          	for(int p = 0; p<PARA; p++)
	    	    	for(int im = 0; im<img_num; im++)
	    	    	{
						float max = 0;
						float idx = 0;
						float sum = 0;
	            		for(int m = 0; m<ksize; m++)
	              			for(int n = 0; n<ksize; n++)
	             			{
	               				float ip = in_buf[m*ksize+n][p*img_num+im];
	               				if(maxpool)
	               				{
									if(ip>max)
	                 				{
	            	   					idx = m*ksize+n;
	            	   					max = ip;
	                 				}
	               				}
	               				else 
	            	   				sum += ip;
	             			}

	             		int out_off = ((y*x_out+x)*outChannel + o*PARA + p)*img_num+im;

	             		if(maxpool)
	             		{
	             			output[out_off] = max;
	             			pos[out_off] = idx;
	             		}
	             		else
	             			output[out_off] = sum/ksize/ksize;

	    	   		}
	        }
}

void back_pool(float *indiff, float *pos, float *outdiff0, float *outdiff1, int *params, int *sparams)
{
	float out_buf[PARA][img_num];
	float out0_buf[PARA][img_num];
	float pos_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int dup = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	int x_out = ((x_dim - ksize) / stride) + 1;
    int y_out = x_out;

 	for(int o = 0; o<iteration_out; o++)
		for(int y = 0; y < y_out; y++)
  			for(int x = 0; x < x_out; x++)
	 		{
				int out_trans_size = PARA;
				int out_offset = ((y*x_out+x)*outChannel+o*PARA);
				memcpy(out_buf, outdiff0 + out_offset*img_num, out_trans_size*img_num*4);
				if(dup)
				{
					memcpy(out0_buf, outdiff1 + out_offset*img_num, out_trans_size*img_num*4);
				}

				memcpy(pos_buf, pos + out_offset*img_num, out_trans_size*img_num*4);

        		#pragma omp parallel for
				for(int p = 0; p<PARA; p++)
		 		for(int im = 0 ;im<img_num ;im++)	
				{
					float tmpout;
					if(dup)
					{
						tmpout = out_buf[p][im] + out0_buf[p][im];          
					}
					else
						tmpout = out_buf[p][im];

					if(maxpool)
					{ 	
						float tmp[64];
						memset_float(tmp, 64);
						int idx = int(pos_buf[p][im]);
						tmp[idx] += tmpout;

        				for(int m = 0; m<ksize; m++)
           					for(int n = 0; n<ksize; n++)
            				{
								int x_in = x*stride+n;
								int y_in = y*stride+m;
								int in_off = ((y_in*x_dim+x_in)*inChannel+o*PARA+p)*img_num+im;
								float tmp_int;
								if (((stride + n) < ksize && x!= 0) || ((stride + m) < ksize && y!=0))
									tmp_int = indiff[in_off] + tmp[m*ksize+n];
								else
									tmp_int = tmp[m*ksize+n];
									
								indiff[in_off] = tmp_int;
							}
					}
					else
						for(int m = 0; m<ksize; m++)
							for(int n = 0; n<ksize; n++)
							{
								int x_in = x*stride+n;
								int y_in = y*stride+m;
								int in_off = ((y_in*x_dim+x_in)*inChannel+o*PARA+p)*img_num+im;
								float tmp_int;
								if (((stride + n) < ksize && x!= 0) || ((stride + m) < ksize && y!=0))
									tmp_int = indiff[in_off] + tmpout/ksize/ksize;
								else
									tmp_int = tmpout/ksize/ksize;  
									
								indiff[in_off] = tmp_int;
							}
				}
	 		}
}
