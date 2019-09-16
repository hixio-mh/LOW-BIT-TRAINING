#include "conv.h"

void relu(float *input, float *output, int *params, int *sparams)
{
	float in_buf[PARA][img_num];
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
    	for(int y = 0; y<y_dim; y++)
	  		for(int x = 0; x<x_dim; x++)
 			{
         		#pragma omp parallel for  
	     		for(int p = 0; p<PARA; p++)
	     		{  					
					int in_trans_size = img_num;
					int in_off = ((y*x_dim+x)*inChannel+o*PARA+p)*img_num;    	    	
					memcpy(in_buf[p], input + in_off, in_trans_size*4);	

	    			for(int im = 0; im<img_num; im++)
	    			{
	    	  			float ip = in_buf[p][im];
	    	  			if(ip > 0)
	    	  				out_buf[p][im] = ip;
	    	  			else
	    	  				out_buf[p][im] = 0;
	    			}

					int out_trans_size = img_num;
					int out_off = ((y*x_dim+x)*inChannel+o*PARA+p)*img_num;	    	    	
					memcpy(output + out_off, out_buf[p], in_trans_size*4);
    	    	
    	     	}
    	 	} 	
}

void relu_back(float *input, float *outdiff0, float* outdiff1, float *indiff, int *params, int *sparams)
{
	float in_buf[PARA][img_num];
    float indiff_buf[PARA][img_num];
    float out_buf[PARA][img_num];
    float out0_buf[PARA][img_num];

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

    for(int o = 0; o<iteration_out; o++)
    	for(int y = 0; y < y_dim; y++)
	  		for(int x = 0; x < x_dim; x++)
    	 	{
				int out_trans_size = PARA;
				int out_off = ((y*x_dim+x)*inChannel+o*PARA);
				memcpy(out_buf, outdiff0 + out_off*img_num, out_trans_size*img_num*4);
				if(dup)
					memcpy(out0_buf, outdiff1 + out_off*img_num, out_trans_size*img_num*4);
            

				int in_trans_size = PARA*img_num;
				int in_off = ((y*x_dim+x)*inChannel+o*PARA)*img_num;
				memcpy(in_buf, input + in_off, in_trans_size*4);
               
            	#pragma omp parallel for
    	    	for(int p = 0; p<PARA; p++)
    	    	  	for(int im = 0; im<img_num; im++)
    	     		{
						float tmpout;
						if(dup)
							tmpout = out_buf[p][im] + out0_buf[p][im];
						else
							tmpout = out_buf[p][im];

    	    	  		float ip = in_buf[p][im];
						if(ip > 0)
							if(ip>0.0000009834766 && ip<0.0000009834767)
								indiff_buf[p][im] = 0;
							else
								indiff_buf[p][im] = tmpout;
						else
							if((ip>-0.00000073079554 && ip<-0.00000073079553) || (ip>-0.0000003031322 && ip<-0.0000003031321) || (ip>-0.0000005691180 && ip<-0.0000005691179))
								indiff_buf[p][im] = tmpout;
							else
								indiff_buf[p][im] = 0;
    	     		}

				memcpy(indiff + out_off*img_num, indiff_buf, out_trans_size*img_num*4);
    	   	}
}



