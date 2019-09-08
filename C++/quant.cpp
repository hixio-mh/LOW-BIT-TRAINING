#include "conv.h"

#define PARA 10


void memset_float(float* to, int size)
{
  	for(int i = 0; i<size; i++)
    	to[i] = 0;
}

float input_qt(float input, int s_in, int *larger, int *smaller, int *all, bool stochastic)
{
	float rand_num = 0;
	float tmp_int;
	if(stochastic){
		rand_num = rand() % 4096 / 4096; 
		tmp_int = floor(input * (1<<s_in) + rand_num);
		if(tmp_int >= 32768 || tmp_int < -32768)
			*larger += 1;
    	if(tmp_int >= 16384 || tmp_int < -16384)
        	*smaller += 1;

    	if(tmp_int > 32767)
       		tmp_int = 32767;
    	else if (tmp_int < -32768)
       		tmp_int = -32768;

		*all += 1;
	}
	else{
		tmp_int = floor(input * (1<<s_in) + 0.5);
   	 		
    	if(tmp_int >= 128 || tmp_int < -128)
			*larger += 1;
    	if(tmp_int >= 64 || tmp_int < -64)
        	*smaller += 1;

    	if(tmp_int > 127)
       		tmp_int = 127;
    	else if (tmp_int < -128)
       		tmp_int = -128;

		*all += 1;
	}
    return  1.0 * tmp_int / (1<<s_in);
}

void quantize_backward(float *to, int size, int *step, bool if_stochastic)
{
	int larger = 0;
	int smaller = 0;
	int all = 0;

  	for(int i = 0; i<size; i++)
    	to[i] = input_qt(to[i], *step, &larger, &smaller, &all, if_stochastic);
    	
	if(!TEST_MODE){ 
	if(1.0*larger/all > 0)
		*step -= 1;
	if(1.0*smaller/all <= 0)
		*step += 1; 
	} 
	
	if(SHOW_QUANTIZE_RESULT)
		printf("%d %f %f\n", *step, 1.0*larger/all, 1.0*smaller/all); 
}

void quantize_forward(float *from, float *to, int size, int *step, bool if_stochastic)
{
	int larger = 0;
	int smaller = 0;
	int all = 0;

  	for(int i = 0; i<size; i++)
    	to[i] = input_qt(from[i], *step, &larger, &smaller, &all, if_stochastic);

	if(!TEST_MODE){ 
	if(1.0*larger/all > 0)
		*step -= 1;
	if(1.0*smaller/all <= 0)
		*step += 1; 
	} 
	
	if(SHOW_QUANTIZE_RESULT)
		printf("%d %f %f\n", *step, 1.0*larger/all, 1.0*smaller/all); 
}

