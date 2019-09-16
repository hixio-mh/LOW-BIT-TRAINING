#include "conv.h"

#define PARA 10

void conv(float *input, float *weights, float* tmp, float *output, int *params, int *sparams){

	float weight_buf[PARA][1024];
	float in_buf[49][PARA*img_num];
    float tmp_buf[PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = x_dim;
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int bias_en = params[9];

	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
	int y_out = x_out;

  	for(int o = 0; o<iteration_out; o++)
   	{
    	for(int i = 0; i<iteration_in; i++)
    	{
       		#pragma omp parallel for
       		for(int p = 0; p<PARA; p++)
			{
				int weight_trans_size = inCh_once*ksize*ksize;
				int weight_offset = (((o*PARA+p))*inChannel+i*inCh_once)*ksize*ksize;
				memcpy(weight_buf[p], weights + weight_offset, weight_trans_size*4);
			}

			for(int y = 0; y < y_out; y++)
				for(int x = 0; x < x_out; x++)
				{
					for(int p = 0; p<ksize; p++)
						for(int q = 0; q<ksize; q++)
						{
							int x_in = x*stride-pad+q;
							int y_in = y*stride-pad+p;
							int inoff = p*ksize+q;
							int insize = inCh_once*img_num;
							if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
							{
								memset_float(in_buf[inoff], insize);
							}
							else if((stride + q < ksize) && x != 0)
							{
								memcpy(in_buf[inoff], in_buf[p*ksize+stride+q], insize*4);
							}
							else
							{
								int in_off = ((y_in*x_dim+x_in)*inChannel+i*inCh_once)*img_num;
								memcpy(in_buf[inoff], input + in_off, insize*4);
							}
						}

					#pragma omp parallel for
					for(int p = 0; p<PARA; p++)
					{
						int tmp_offset = ((y*x_out+x)*PARA+p)*img_num;
						memcpy(tmp_buf[p], tmp + tmp_offset, 4*img_num);
						for(int in = 0; in<inCh_once; in++)
							for(int m = 0; m<ksize; m++)
								for(int n = 0; n<ksize; n++)
								{
									float wg = weight_buf[p][in*ksize*ksize+m*ksize+n];
									for(int im = 0; im<img_num; im++)
									{
										if (i == 0 && in == 0 && m==0 && n==0)
											tmp_buf[p][im] = in_buf[m*ksize+n][in*img_num + im]*wg;
										else
											tmp_buf[p][im] += in_buf[m*ksize+n][in*img_num + im]*wg;
									}
								}


						if(i == iteration_in - 1)
						{
							for(int im = 0; im<img_num; im++)     
							{
								int out_offset = ((y*x_out+x)*outChannel+o*PARA+p)*img_num+im;
								output[out_offset] = tmp_buf[p][im];
							}
						}
						else 
							memcpy(tmp + tmp_offset, tmp_buf[p], 4*img_num);
					}
				}
    	}
    }
}

#define PARA 1

void conv_back(float *outdiff, float *weights, float *indiff, float *tmp, int *params, int *sparams)
{
	float weight_buf[PARA][1024];
	float out_buf[64][img_num];
  	float in_buf[49][PARA][img_num];

	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/PARA;
	int iteration_out = outChannel/outCh_once;
	int x_dim = params[4];
	int y_dim = x_dim;
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
  	int y_out = x_out;

  	for(int i = 0; i<iteration_in; i++)
    	for(int o = 0; o<iteration_out; o++)
    	{
      		#pragma omp parallel for
       		for(int p = 0; p<outCh_once; p++)
       		{
				int weight_trans_size = ksize*ksize;
				int weight_offset = ((o*outCh_once+p)*inChannel+i*PARA)*ksize*ksize;
    	 		for(int m = 0; m<PARA; m++)
        		{      
		    		memcpy(weight_buf[m] + ksize*ksize*p, weights + weight_offset + m*ksize*ksize, weight_trans_size*4);
        		} 
       		}


			for(int y = 0; y < y_out; y++)
				for(int x = 0; x < x_out; x++)
    	 		{
    				int out_trans_size = outCh_once*img_num;
    				int out_offset = ((y*x_out + x)*outChannel+o*outCh_once)*img_num;
    				memcpy(out_buf, outdiff + out_offset ,out_trans_size*4);

        			for(int p=0;p<ksize;p++)
          				for(int q=0;q<ksize;q++)
        				{
							int x_in = x*stride-pad+q;
							int y_in = y*stride-pad+p;
							int inoff = p*ksize+q;
							int insize = PARA*img_num;
							if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
							{
							}
							else if((stride + q < ksize) && x != 0)
							{
								memcpy(in_buf[inoff], in_buf[p*ksize+stride+q], insize*4);
							}
							else
							{
							int in_off = ((y_in*x_dim+x_in) * PARA)*img_num;
								memcpy(in_buf[inoff], tmp + in_off, insize*4);
							}
						}

         			#pragma omp parallel for
         			for(int p = 0; p<PARA; p++)
         			{
          				for(int on = 0;on<outCh_once;on++)
           					for(int m = 0; m<ksize; m++)
            					for(int n = 0; n<ksize; n++)
           						{
        	   						float wg = weight_buf[p][on*ksize*ksize+m*ksize+n];
									for(int im = 0 ; im<img_num ;im++) 
									{
										if(o==0 && on==0  &&  !((stride + n < ksize) && (x != 0)) &&  !((stride + m < ksize) && (y != 0)))
											in_buf[m*ksize+n][p][im] = wg*out_buf[on][im];
										else
											in_buf[m*ksize+n][p][im] += wg*out_buf[on][im];
           							}
    	     					}
          			}

					for(int m = 0; m<ksize; m++)
						for(int n = 0; n<ksize; n++)
						{
							int x_in = x*stride-pad+n;
							int y_in = y*stride-pad+m;

          					if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
          						continue;
							int in_trans_size = PARA;
							int tmp_offset = (y_in * x_dim+x_in)*PARA*img_num;
							int in_offset =  ((y_in * x_dim+x_in)*inChannel+i*PARA)*img_num;
          					if ((o == iteration_out - 1) && !((( n - stride >= 0 ) && 
							  (x != x_out -1 )) || (( m - stride >= 0) && (y != y_out - 1))))
            				{ 
								#pragma omp parallel for             
								for(int p = 0; p<PARA ; p++)
								{ 
              						for(int im = 0; im<img_num ;im++) 
              						{
              							indiff[in_offset+p*img_num+im] = in_buf[m*ksize+n][p][im];
              						}
             			 		}
            				}
          					else  
       	    					memcpy(tmp + tmp_offset, in_buf[m*ksize+n], PARA*img_num*4);
          				}

        		}
     	}
}

#define PARA 10

void weight_back(float *outdiff, float *weights, float *input, float * moment, int *params, int *sparams, int rate)
{
	int inChannel = params[0];
	int outChannel = params[1];
	int inCh_once = params[2];
	int outCh_once = params[3];
	int iteration_in = inChannel/inCh_once;
	int iteration_out = outChannel/PARA;
	int x_dim = params[4];
	int y_dim = x_dim;
	int stride = params[6];
	int pad = params[7];
	int ksize = params[8];
	int maxpool = params[9];

	float in_buf[ksize*ksize][inCh_once*img_num];
	float weight_buf[PARA][inCh_once*ksize*ksize];
	float moment_buf[PARA][inCh_once*ksize*ksize];
	float out_buf[PARA][img_num];

	int x_out = ((x_dim - ksize + 2 * pad) / stride) + 1;
	int y_out = x_out;

   	for(int i = 0; i<iteration_in; i++)
    	for(int o = 0; o<iteration_out; o++)
    	{
    		for(int y = 0; y < y_out; y++)
    	  		for(int x = 0; x < x_out; x++)
    	 		{
    	 			ksize = params[8];
					int out_trans_size = PARA*img_num;
					int out_offset = ((y*x_out+x)*outChannel+o*PARA)*img_num;
					memcpy(out_buf, outdiff + out_offset ,out_trans_size*4);

    				for(int p = 0; p<ksize; p++)
			   			for(int q = 0; q<ksize; q++)
			   			{
							int x_in = x*stride-pad+q;
							int y_in = y*stride-pad+p;
							int inoff = (p*ksize+q);
							int insize = inCh_once*img_num;
							if(x_in<0 || y_in<0 || x_in>x_dim-1 || y_in>y_dim-1)
							{
								memset_float(in_buf[inoff], insize);
							}
							else if((stride + q < ksize) && x != 0)
							{
								memcpy(in_buf[inoff], in_buf[p*ksize+stride+q], insize*4);
							}
							else
							{
								int in_off = ((y_in*x_dim+x_in)*inChannel+i*inCh_once)*img_num;
								memcpy(in_buf[inoff], input + in_off, insize*4);
							}
			   			}

					#pragma omp parallel for
					for(int p = 0; p<PARA; p++)
					{
						for(int in = 0;in<inCh_once;in++)
							for(int m = 0; m<ksize; m++)
								for(int n = 0; n<ksize; n++)
								{
									for(int im=0;im<img_num;im++)
									{
        								if(x==0 && y==0 && im==0)
        									weight_buf[p][in*ksize*ksize+m*ksize+n] = out_buf[p][im]*in_buf[m*ksize+n][in*img_num+im];
        								else
        									weight_buf[p][in*ksize*ksize+m*ksize+n] += out_buf[p][im]*in_buf[m*ksize+n][in*img_num+im];
    	      						}
        						}
    	   			}
        		}    	 

			#pragma omp parallel for
			for(int p = 0; p<PARA; p++)
			{
				int weight_trans_size = inCh_once*ksize*ksize;
				int weight_offset = ((o*PARA+p)*inChannel+i*inCh_once)*ksize*ksize;
				memcpy(moment_buf[p], moment + weight_offset, weight_trans_size*4);

      	  		for (int l = 0; l<weight_trans_size; l++)
      	  		{
					moment_buf[p][l] = 0.9*moment_buf[p][l] + weight_buf[p][l]/rate;
					weights[weight_offset+l] = weights[weight_offset+l] - moment_buf[p][l]; 
          		}
          
         		memcpy(moment + weight_offset, moment_buf[p], weight_trans_size*4);
        	}

    	}
}
