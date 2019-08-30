#include "conv.h"

#define PARA 10

void fc(float *input, float *weights, float  *bias, float  *output, int *params, int *sparams){

	float weight_buf[PARA][64];
	float in_buf[64][img_num];
  	float bias_buf[PARA];
  	float out_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

    for(int o = 0; o<iteration_out; o++)
    {   
		int bias_offset = o*PARA;
		memcpy(bias_buf, bias + bias_offset , PARA*4);

    	#pragma omp parallel for      
     	for(int p = 0; p<PARA; p++)
      		for(int im = 0; im<img_num; im++)
      		{
          		out_buf[p][im] = bias_buf[p];
      		}

	    for(int i = 0; i<iteration_in; i++)
	     	for(int y = 0; y < y_dim; y++)
    		{ 
         		for(int x = 0; x < x_dim; x++)
    	 		{
					int in_trans_size = inCh_once*img_num;
					int in_offset = ((y*x_dim+x)*inChannel+i*inCh_once)*img_num;
					memcpy(in_buf, input + in_offset, in_trans_size*4);

        			#pragma omp parallel for
    	   			for(int p = 0; p<PARA; p++)
    	    		{
						int wg_trans_size = inCh_once;
						int wg_offset = ((y*x_dim+x)*outChannel+o*PARA+p)*inChannel+i*inCh_once;
						memcpy(weight_buf[p], weights + wg_offset, wg_trans_size*4);
    	    		}

    				#pragma omp parallel for
    	   			for(int p = 0; p<PARA; p++)
    	   			{
    		  			for(int in = 0; in<inCh_once; in++)
    		  			{ 
            				float  wg = weight_buf[p][in];
            				for(int im = 0; im<img_num; im++)
								out_buf[p][im] += wg*in_buf[in][im];
    	    			}

         			}
    	 		}  
			}
   
       	int out_offset = o*PARA*img_num;
        for(int p = 0; p<PARA; p++)
          	for(int im = 0; im < img_num; im++)
          	{
				output[out_offset+p*img_num+im] = out_buf[p][im];
          	}
       
    }
}

#define PARA 20

void back_fc(float  *outdiff, float *weights, float *tmp, float *indiff, int *params, int *sparams){

	float  weight_buf[PARA][64];
	float  out_buf[64][img_num];
  	float  in_buf[PARA][img_num];
    int flag = -1;

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/outCh_once;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

    for(int o = 0; o<iteration_out; o++)
    {   
      	int out_trans_size = outCh_once*img_num;
	    int out_offset = o*outCh_once*img_num;
      	memcpy(out_buf, outdiff + out_offset, out_trans_size*4);

	    for(int i = 0; i<iteration_in; i++)
	     	for(int y = 0; y < y_dim; y++)
    	  		for(int x = 0; x < x_dim; x++)
    	   		{
					int in_trans_size = PARA*img_num;
					int in_offset = (y*x_dim+x)*PARA;
					memcpy(in_buf, tmp + in_offset*img_num, 4*in_trans_size);   
    	   	
    	   			#pragma omp parallel for
    	   			for(int on = 0; on<outCh_once; on++)
    	    		{
             			int wg_offset = ((y*x_dim+x)*outChannel+o*outCh_once+on)*inChannel+i*PARA;
             			for(int p = 0; p<PARA; p++)
                			weight_buf[p][on] = weights[wg_offset + p];
    	    		}

					#pragma omp parallel for
					for(int p = 0; p<PARA; p++)
					{
						for(int on = 0; on<outCh_once; on++)
						{ 
            				float  wg = weight_buf[p][on];
          					for(int im =0 ; im< img_num; im++)
							{    	    
								if(on == 0 && o == 0)
             						in_buf[p][im] = wg*out_buf[on][im];
            					else
             						in_buf[p][im] += wg*out_buf[on][im];
  							}

          				}    	        
         			} 

					if(o == iteration_out - 1)
					{
						for(int p = 0; p<PARA; p++)
          					for(int im = 0 ;im<img_num ;im++) 
         					{
								int in_offset = ((y*x_dim+x)*inChannel+i*PARA + p)*img_num+im;
								indiff[in_offset] = in_buf[p][im];
          					}
        			}
        			else 
        			{ 
						int tmp_offset = (y*x_dim+x)*PARA*img_num;
						memcpy(tmp + tmp_offset, in_buf, 4*PARA*img_num);
         			}
    	 		}
    }
}

#define PARA 10

void fc_weight(float *outdiff, float *weights, float *input, float *moment, int *params, int *sparams, int rate)
{
	float weight_buf[PARA][64];
  	float moment_buf[PARA][64];
	float in_buf[64][img_num];
  	float out_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

    for(int o = 0; o<iteration_out; o++)
    {   
      	int out_trans_size = PARA*img_num;
	    int out_offset = o*PARA;
	    memcpy(out_buf, outdiff + out_offset*img_num, out_trans_size*4);

	    for(int i = 0; i<iteration_in; i++)
	     	for(int y = 0; y < y_dim; y++)
    	  		for(int x = 0; x < x_dim; x++)
    	  		{
					int in_trans_size = inCh_once*img_num;
					int in_offset = ((y*x_dim+x)*inChannel+i*inCh_once)*img_num;
					memcpy(in_buf, input + in_offset, in_trans_size*4);
            
					#pragma omp parallel for
					for(int p = 0; p<PARA; p++)
					{
						int wg_offset = ((y*x_dim+x)*outChannel+o*PARA+p)*inChannel+i*inCh_once;
						memcpy(moment_buf[p], moment + wg_offset, inCh_once*4);
    	     		}
  		    
					#pragma omp parallel for
					for(int p = 0; p<PARA; p++)
					{
    		    		for(int in = 0; in<inCh_once; in++)
    		    		{
							int wg_offset = ((y*x_dim+x)*outChannel+o*PARA+p)*inChannel+i*inCh_once+in;
							weight_buf[p][in] = 0;

    		    			for(int im = 0; im<img_num; im++)
    	         				weight_buf[p][in]+= in_buf[in][im]*out_buf[p][im];


		      				moment_buf[p][in] = 0.9*moment_buf[p][in] + weight_buf[p][in]/rate;
          					weights[wg_offset] = weights[wg_offset] - moment_buf[p][in];
		       			}

    	     		}

    		  		#pragma omp parallel for
           			for(int p =0; p<PARA;p++)
            		{
						int wg_offset = ((y*x_dim+x)*outChannel+o*PARA+p)*inChannel+i*inCh_once;
						memcpy(moment + wg_offset, moment_buf[p], inCh_once*4);
            		}

    	   		}
    }
}


void bias_back_fc(float *outdiff, float *bias, float * moment, int *params, int *sparams, int rate)
{
	float bias_buf[PARA];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = params[5];
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
  	int y_out = x_out;

	for(int i = 0; i<iteration_out; i++)
	{
		#pragma omp parallel for
		for(int o = 0; o< PARA; o++)
		{
    		int bias_offset = (i*PARA+o);
        	float moment_buf = moment[bias_offset];

        	float tmp0 = 0;
        	for(int im = 0; im<img_num ;im++)
           		tmp0 += outdiff[bias_offset*img_num+im]; 

        	moment_buf = 0.9*moment_buf + tmp0/rate;
        	bias[bias_offset] = bias[bias_offset] - moment_buf;
        	moment[bias_offset] = moment_buf;
        }
    }
}


