#include "conv.h"

#define PARA 10

#define layer_num 8

void training(float *data, float *wg, float *fp, int *setting, int *sp, int rate, int epoch, int decay)
{
    int params[layer_num][16];
    int offset[layer_num][10];
    int connect[layer_num][4];
    float tmp[250880]; 
    
    int (*sparams)[8] = (int (*)[8])  sp; 
	for(int i = 0; i<layer_num; i++)
	{
		memcpy(params[i], setting+30*i, 16*4);
		memcpy(offset[i], setting+30*i+16, 10*4);
		memcpy(connect[i], setting+30*i+26, 4*4);
	}

	for(int j = 0; j<layer_num; j++ )
	{
       switch(params[j][10])
       {
       		case 0:
				if(CONV_FORWARD){
					printf("conv_input_q : ");
					quantize(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

				if(CONV_WEIGHT){
					printf("conv_weight_q : ");
					quantize(data+offset[j][1], params[j][0]*params[j][1]*params[j][8]*params[j][8], &sparams[j][4], false);
				}

             	if(SHOW_CONV_IN) 
             	{  
					printf("conv_input:\n");
                	for(int y = 0; y < 28; y++) 
                	{
						for(int x = 0; x < 28; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+(y*28+x)*img_num)));
                		printf("\n"); 
                	}
              	}

    	        conv(data+offset[j][0], data+offset[j][1], tmp, data+offset[j][4], params[j], sparams[j]);
               	break;

       		case 1: 
			   	if(BATCH_FORWARD){
					printf("batch_input_q : ");
					quantize(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

				if(SHOW_BATCH_IN)
				{ 
				   	printf("batch_input:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+((y*24+x)*20)*img_num)));
                		printf("\n");
					}
				}

    	        batch(data+offset[j][0], wg+offset[j][1], data+offset[j][4], params[j], sparams[j]);
    	        break;

       		case 2:  
			   	if(SCALE_FORWARD){
					printf("scale_input_q : ");
					quantize(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

			   	if(SCALE_WEIGHT){
					printf("scale_weight_q : ");
					quantize(data+offset[j][1], params[j][1], &sparams[j][4], false);
				}

			   	if(SCALE_BIAS){
					printf("scale_bias_q : "); 
					quantize(data+offset[j][3], params[j][1], &sparams[j][5], false);
				}

              	if(SHOW_SCALE_IN) 
              	{  
					printf("scale_input:\n");
                	for(int y = 0; y < 24; y++) 
           			{     
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+((y*24+x))*20*img_num)));
                		printf("\n"); 
        			}
				}

    	        scale(data+offset[j][0], data+offset[j][1], data+offset[j][3], data+offset[j][4], params[j], sparams[j]);
    	        break;

       		case 3:  
              	if(SHOW_RELU_IN) 
               	{ 
					printf("relu_input:\n");
                	for(int y = 0; y < 24; y++) 
         			{       		
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+((y*24+x)*20)*img_num)));
                		printf("\n"); 
    	  			} 
				}

               	relu(data+offset[j][0], data+offset[j][4], params[j], sparams[j]);
                break;

       		case 4:  
              	if(SHOW_POOL_IN) 
              	{  
					printf("pool_input:\n");
                	for(int y = 0; y < 24; y++) 
           			{     
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+(y*24+x)*20*img_num)));
                		printf("\n"); 
    	    		}  
				}  

               	pool(data+offset[j][0], data+offset[j][1], data+offset[j][4], params[j], sparams[j]);
            	break;

       		case 5:  
			   	break;

       		case 6:  
			   	if(FULL_FORWARD){
					if(j == 5)
						printf("fc1_input_q : ");
					else
						printf("fc2_input_q : ");
					quantize(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

			   	if(FULL_WEIGHT){
					if(j == 5)
						printf("fc1_weight_q : ");
					else
						printf("fc2_weight_q : ");
					quantize(data+offset[j][1], params[j][0]*params[j][1]*params[j][4]*params[j][5], &sparams[j][4], false);
				}

			   	if(FULL_BIAS){
					if(j == 5)
						printf("fc1_bias_q : ");
					else
						printf("fc2_bias_q : ");
					quantize(data+offset[j][3], params[j][1], &sparams[j][5], false);
				}

              	if(SHOW_FULL_IN) 
              	{  
                	if(j == 5){
						printf("fc1_input:\n");
                		for(int y = 0; y < 12; y++) 
             			{   
							for(int x = 0; x < 12; x++) 
                				printf(" %f", (float)(*(data+offset[j][0]+((y*12+x)*20)*img_num)));
               				printf("\n"); 
               			}
					}
					if(j == 6) 
					{ 
						printf("fc2_input:\n");
						for(int y = 0; y < 100; y++) 
							printf(" %f", (float)(*(data+offset[j][0]+(y*img_num)))); 
					}
				} 

    	        fc(data+offset[j][0], data+offset[j][1], data+offset[j][3], data+offset[j][4], params[j], sparams[j]);
    	        break;
    	        
       		case 7:  
              	if(SHOW_SOFTMAX_IN) 
              	{  
					printf("softmax_input:\n");
                	for(int i = 0; i < 10; i++) 
                		printf(" %f", (float)(*(data+offset[j][0]+i*img_num)));
                	printf("\n");  
				}
    	        softmax(data+offset[j][0], data+offset[j][4], params[j], sparams[j], data+offset[j][2], (float*)(wg+offset[j][1]));
       	}
 	}
     
	for(int j = layer_num-2; j>=0; j--)
	{
		switch(params[j][10])
		{
			case 0: 
				if(CONV_BACKWARD){
					printf("conv_outdiff_q : ");
					quantize(data+offset[j][5], 24*24*params[j][1]*img_num, &sparams[j][2], true);
				}

               	if(SHOW_BATCH_INDIFF)
               	{ 
					printf("batch_indif:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][5]+((y*24+x)*20)*img_num)));
                		printf("\n");
					}
				}
                
				if(SHOW_CONV_WEIGHT)
               	{ 
					printf("conv_weight_before_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+(2*5+1)+y*25)));
                	printf("\n");
				}

			    conv_back(data + offset[j][5], data+offset[j][1], data+offset[j][2], tmp, params[j], sparams[j]);
		        weight_back(data + offset[j][5], data+offset[j][1], data+offset[j][0], wg+offset[j][6], params[j], sparams[j], rate);

                if(SHOW_CONV_INDIFF)
               	{ 
					printf("conv_indif:\n");
                	for(int y = 0; y < 28; y++) 
               		{ 
						for(int x = 0; x < 28; x++) 
                			printf(" %f", (float)(*(data+offset[j][2]+((y*28+x)*1)*img_num)));
                		printf("\n");
					}
				}
                
				if(SHOW_CONV_WEIGHT)
               	{ 
					printf("conv_weight_after_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+(2*5+1)+y*25)));
                	printf("\n");
				}
                break;

			case 1: 
				if(BATCH_BACKWARD){
					printf("batch_outdiff_q : ");
					quantize(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], true);
				}

               	if(SHOW_SCALE_INDIFF)
               	{ 
					printf("scale_indif:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
							printf(" %f", (float)(*(data+offset[j][5]+(y*24+x)*20*img_num)));
						printf("\n");
					}
				}
			    batch_back(data+ offset[j][4], data+ offset[j][2], wg+offset[j][1], data+ offset[j][5], params[j], sparams[j]);
                break;

			case 2:  
				if(SCALE_BACKWARD){
					printf("scale_outdiff_q : ");
					quantize(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], true);
				}

               	if(SHOW_RELU_INDIFF)
               	{ 
				   	printf("relu_indif:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][5]+(y*24+x)*20*img_num)));
                		printf("\n");
					}
				}

               	if(SHOW_SCALE_WEIGHT)
               	{ 
				   	printf("scale_weight_before_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+y)));
                	printf("\n");

				}

               	if(SHOW_SCALE_BIAS)
               	{ 
				   	printf("bias_weight_before_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+y)));
                	printf("\n");

				}
				
				scale_back(data+offset[j][2], data+offset[j][1], data+offset[j][5], params[j], sparams[j]);
				weight_scale(data+offset[j][0], data+offset[j][1], data+offset[j][5],  wg+offset[j][6], params[j], sparams[j], rate);
				bias_scale(data+offset[j][3], data+offset[j][5], wg+offset[j][7], params[j], sparams[j], rate);
                
               	if(SHOW_SCALE_WEIGHT)
               	{ 
				   	printf("scale_weight_after_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+y)));
                	printf("\n");

				}
				
               	if(SHOW_SCALE_BIAS)
               	{ 
				   	printf("scale_bias_before_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][1]+y)));
                	printf("\n");

				}
				break;

			case 3: 
               if(SHOW_POOL_INDIFF)
               	{ 
				   	printf("pool_indif:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][5]+(y*24+x)*20*img_num)));
                		printf("\n");
					}
				}

			    relu_back(data+offset[j][0], data+offset[j][5], data+offset[j][6], data+offset[j][2], params[j], sparams[j]);
		        break;

			case 4: 
               if(SHOW_FULL_INDIFF)
               	{ 
				   	printf("full1_indif:\n");
                	for(int y = 0; y < 12; y++) 
               		{ 
						for(int x = 0; x < 12; x++) 
                			printf(" %f", (float)(*(data+offset[j][5]+((y*12+x)*20)*img_num)));
                		printf("\n");
					}
				}

			    back_pool(data+offset[j][2], data+offset[j][1], data+offset[j][5], data+offset[j][6], params[j], sparams[j]);
			    break;

			case 5: 
				break;

			case 6: 
				if(FULL_BACKWARD){
					if(j == 5)
						printf("fc1_outdiff_q : ");
					else
						printf("fc2_outdiff_q : ");
					quantize(data+offset[j][5], params[j][1]*img_num, &sparams[j][2], true);
				}

              	if(SHOW_FULL_INDIFF && j == 5) 
              	{  
					printf("full2_indif:\n");
                	for(int x = 0; x < 100; x++) 
                		printf(" %lf", (float)(*(data+offset[j][5]+x*img_num)));
               		printf("\n"); 
               	}

                if(SHOW_SOFTMAX_INDIFF && j == 6) 
             	{   
					printf("softmax_indif:\n");
					for(int x = 0; x < 10; x++ ) 
                		printf(" %lf", (float)(*(data+offset[j][5]+x*img_num)));
               		printf("\n"); 
				}

        		if(j == 5){
					if(SHOW_FULL1_WEIGHT){
						printf("fc1_weight_before_backward:\n");
						for(int x = 0; x < 100; x++)
							printf(" %f", (float)(*(data+offset[j][1]+x*20)));
						printf("\n");
					}
					if(SHOW_FULL1_BIAS){
						printf("fc1_bias_before_backward:\n");
						for(int x = 0; x < 100; x++)
							printf(" %f", (float)(*(data+offset[j][3]+x)));
						printf("\n");	
					}
				}

        		if(j == 6){
					if(SHOW_FULL2_WEIGHT){
						printf("fc2_weight_before_backward:\n");
						for(int x = 0; x < 10; x++)
							printf(" %f", (float)(*(data+offset[j][1]+x*100)));
						printf("\n");
					}
					if(SHOW_FULL2_BIAS){
						printf("fc2_bias_before_backward:\n");
						for(int x = 0; x < 10; x++)
							printf(" %f", (float)(*(data+offset[j][3]+x)));
						printf("\n");	
					}
				}

			    back_fc(data+offset[j][5], data+offset[j][1], tmp, data+offset[j][2], params[j], sparams[j]);
			    fc_weight(data+offset[j][5], data+offset[j][1], data+offset[j][0], wg+offset[j][6], params[j], sparams[j], rate);
			    bias_back_fc(data+offset[j][5], data+offset[j][3], wg+offset[j][7], params[j], sparams[j], rate);

				if(j == 5){
					if(SHOW_FULL1_WEIGHT){
						printf("fc1_weight_after_backward:\n");
						for(int x = 0; x < 100; x++)
							printf(" %f", (float)(*(data+offset[j][1]+x*20)));
						printf("\n");
					}
					if(SHOW_FULL1_BIAS){
						printf("fc1_bias_after_backward:\n");
						for(int x = 0; x < 100; x++)
							printf(" %f", (float)(*(data+offset[j][3]+x)));
						printf("\n");	
					}
				}

        		if(j == 6){
					if(SHOW_FULL2_WEIGHT){
						printf("fc2_weight_after_backward:\n");
						for(int x = 0; x < 10; x++)
							printf(" %f", (float)(*(data+offset[j][1]+x*100)));
						printf("\n");
					}
					if(SHOW_FULL2_BIAS){
						printf("fc2_bias_after_backward:\n");
						for(int x = 0; x < 10; x++)
							printf(" %f", (float)(*(data+offset[j][3]+x)));
						printf("\n");	
					}
				}
			    break;

		}           
	}
}







