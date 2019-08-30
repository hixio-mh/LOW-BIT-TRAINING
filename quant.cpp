#include "conv.h"

#define PARA 10


void memset_float(float* to, int size)
{
  	for(int i = 0; i<size; i++)
    	to[i] = 0;
}

int input_qt(float input, int s_in, int *larger, int *smaller, int *all, bool stochastic)
{
	float rand_num = 0;
	if(stochastic)
		rand_num = rand() % 1024 / 1024; 

	int tmp_int = floor(input * (1<<s_in) + rand_num);
   	 		
    if(tmp_int > 127 || tmp_int < -128)
		*larger += 1;
    if(tmp_int < 63 && tmp_int > -64)
        *smaller += 1;

    if(tmp_int > 127)
       	tmp_int = 127;
    else if (tmp_int < -128)
       	tmp_int = -128;

	*all += 1;
    return  tmp_int;
}

void quantize(float *to, int size, int *step, bool if_stochastic)
{
	int larger = 0;
	int smaller = 0;
	int all = 0;

  	for(int i = 0; i<size; i++)
    	to[i] = input_qt(to[i], *step, &larger, &smaller, &all, if_stochastic);

	if(1.0*larger/all > 0)
		*step -= 1;
	if(1.0*smaller/all >= 1)
		*step += 1; 
	printf("%d %f %f\n", *step, 1.0*larger/all, 1.0*smaller/all); 
}


