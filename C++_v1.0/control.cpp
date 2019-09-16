#include "conv.h"

#define PARA 10

#define layer_num 8

void training(float *data, float *wg, float *fp, int *setting, int *sp, int rate, int epoch, int decay, float *temp_weight, float *temp_bias, int flag)
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
       			
       			if(SHOW_CONV_IN) 
             	{  
					printf("conv_input:\n");
                	for(int y = 0; y < 28; y++) 
                	{
						for(int x = 0; x < 28; x++) {
                			printf(" %f", (float)(*(data+offset[j][0]+(y*28+x)*img_num)));
                		}
                		printf("\n"); 
                	}
              	}
				
				if(CONV_FORWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("conv_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

				if(CONV_WEIGHT){
					if(SHOW_QUANTIZE_RESULT)
						printf("conv_weight_q : ");
					quantize_forward(data+offset[j][1], wg+offset[j][8], params[j][0]*params[j][1]*params[j][8]*params[j][8], &sparams[j][4], false);
				}

				if(0 && flag == 1){
					FILE* conv_in = fopen("conv_in_q.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 28; l++)
							for(int m = 0; m < 28; m++)
								for(int n = 0; n < 1; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][0]+(l*28+m)*img_num+p))); 
				}

				if(0 && flag == 1){
                	FILE* conv_in_w = fopen("conv_w_q.txt","w");
                	for(int p = 0; p < 5; p++)
                		for(int l = 0; l < 5; l++)
                			for(int m = 0; m < 1; m++)
                				for(int n = 0; n < 20; n++)
									fprintf(conv_in_w, "%.6f\n", (float)(*(wg+offset[j][8] + p*5+l+n*25))); 
				}
					
				if (CONV_WEIGHT)
    	        	conv(data+offset[j][0], wg+offset[j][8], tmp, data+offset[j][4], params[j], sparams[j]);
				else
					conv(data+offset[j][0], data+offset[j][1], tmp, data+offset[j][4], params[j], sparams[j]);
               	break;

       		case 1: 
       			if(SHOW_BATCH_IN)
				{ 
				   	printf("batch_input:\n");
                	for(int y = 0; y < 24; y++) 
               		{ 
						for(int x = 0; x < 24; x++) 
                			printf(" %f", (float)(*(data+offset[j][0]+((y*24+x)*20+18)*img_num+14)));
                		printf("\n");
					}
				}
				
       			if(0 && flag == 1){
					FILE* batch_in_q = fopen("batch_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(batch_in_q, "%.6f\n", (float)(*(data+offset[j][0]+((l*24+m)*20+n)*img_num+p))); 
				}
				
			   	if(BATCH_FORWARD){
			   		if(SHOW_QUANTIZE_RESULT)
						printf("batch_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}
              
    	        batch(data+offset[j][0], wg+offset[j][1], data+offset[j][4], params[j], sparams[j]);
    	        
    	        break;

       		case 2:  
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
				
				if(0 && flag == 1){
					FILE* scale_in = fopen("scale_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(scale_in, "%.6f\n", (float)(*(data+offset[j][0]+((l*24+m)*20+n)*img_num+p))); 
				}
				
			   	if(SCALE_FORWARD){
			   		if(SHOW_QUANTIZE_RESULT)
						printf("scale_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

			   	if(SCALE_WEIGHT){
			   		if(SHOW_QUANTIZE_RESULT)
						printf("scale_weight_q : ");
					quantize_forward(data+offset[j][1], wg+offset[j][8], params[j][1], &sparams[j][4], false);
				}

			   	if(SCALE_BIAS){
			   		if(SHOW_QUANTIZE_RESULT)
						printf("scale_bias_q : "); 
					quantize_forward(data+offset[j][3], wg+offset[j][9], params[j][1], &sparams[j][5], false);
				}

				if(SCALE_WEIGHT && SCALE_BIAS)
    	        	scale(data+offset[j][0], wg+offset[j][8], wg+offset[j][9], data+offset[j][4], params[j], sparams[j]);
    	    	else
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

				if(0 && flag == 1){
					FILE* relu_in = fopen("relu_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][0]+((l*24+m)*20+n)*img_num+p))); 
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
				
					if(j == 6) { 
						printf("fc2_input:\n");
						for(int y = 0; y < 100; y++) 
							printf(" %f", (float)(*(data+offset[j][0]+(y*img_num)))); 
					}					
				} 

				if(0 && j == 5 && flag == 1){
					FILE* fc1_in = fopen("fc1_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 12; l++)
							for(int m = 0; m < 12; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(fc1_in, "%.6f\n", (float)(*(data+offset[j][0]+((l*12+m)*20+n)*img_num+p))); 
				}

				if(0 && j==6 & flag == 1){
				    FILE* fc1_in = fopen("fc2_in.txt","w");
                	for(int p = 0; p < 16; p++)
                		for(int l = 0; l < 100; l++)	
							fprintf(fc1_in, "%.6f\n", (float)(*(data+offset[j][0]+l*img_num+p))); 
				}
				
			   	if(FULL_FORWARD){
					if(j == 5)
						if(SHOW_QUANTIZE_RESULT)
							printf("fc1_input_q : ");
					else
						if(SHOW_QUANTIZE_RESULT)
							printf("fc2_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

			   	if(FULL_WEIGHT){
					if(j == 5)
						if(SHOW_QUANTIZE_RESULT)
							printf("fc1_weight_q : ");
					else
						if(SHOW_QUANTIZE_RESULT)
							printf("fc2_weight_q : ");
					quantize_forward(data+offset[j][1], wg+offset[j][8], params[j][0]*params[j][1]*params[j][4]*params[j][5], &sparams[j][4], false);
				}

			   	if(FULL_BIAS){
					if(j == 5)
						if(SHOW_QUANTIZE_RESULT)
							printf("fc1_bias_q : ");
					else
						if(SHOW_QUANTIZE_RESULT)
							printf("fc2_bias_q : ");
					quantize_forward(data+offset[j][3], wg+offset[j][9], params[j][1], &sparams[j][5], false);
				}

				if(FULL_WEIGHT && FULL_BIAS)
    	        	fc(data+offset[j][0], wg+offset[j][8], wg+offset[j][9], data+offset[j][4], params[j], sparams[j]);
    	        else
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

				if(0 && flag == 1){
					FILE* fc1_in = fopen("softmax_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 10; l++)
							fprintf(fc1_in, "%.6f\n", (float)(*(data+offset[j][0]+l*img_num+p))); 
				}

    	        softmax(data+offset[j][0], data+offset[j][4], params[j], sparams[j], data+offset[j][2], (float*)(wg+offset[j][1]));
       	}
 	}
     
	for(int j = layer_num-2; j>=0; j--)
	{
		switch(params[j][10])
		{
			case 0: 
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
				
				if(1 && flag==1){
					FILE* relu_in = fopen("batch_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][5]+((l*24+m)*20+n)*img_num+p))); 
				}
				
				// Important: Change 24 
				if(CONV_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("conv_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], 24*24*params[j][1]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], 24*24*params[j][1]*img_num, &sparams[j][2], true);
				}

				if(CONV_WEIGHT)
			    	conv_back(data + offset[j][5], wg+offset[j][8], data+offset[j][2], tmp, params[j], sparams[j]);
				else
					conv_back(data + offset[j][5], data+offset[j][1], data+offset[j][2], tmp, params[j], sparams[j]);
		        weight_back(data + offset[j][5], data+offset[j][1], data+offset[j][0], wg+offset[j][6], params[j], sparams[j], rate);

				if(1 && flag == 1){
					FILE* relu_in = fopen("conv_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 28; l++)
							for(int m = 0; m < 28; m++)
								for(int n = 0; n < 1; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][2]+((l*28+m)*1+n)*img_num+p))); 
				}
				
				if(0 && flag == 1){
                	FILE* conv_in_w = fopen("conv_w.txt","w");
                	for(int p = 0; p < 5; p++)
                		for(int l = 0; l < 5; l++)
                			for(int m = 0; m < 1; m++)
                				for(int n = 0; n < 20; n++)
									fprintf(conv_in_w, "%.6f\n", (float)(*(data+offset[j][1] + p*5+l+n*25))); 
				}
				
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
				
				if(1 && flag==1){
					FILE* relu_in = fopen("scale_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][5]+((l*24+m)*20+n)*img_num+p))); 
				}
				
				if(BATCH_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("batch_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], true);
				}

			    batch_back(data+ offset[j][4], data+ offset[j][2], wg+offset[j][1], data+ offset[j][5], params[j], sparams[j]);
                break;

			case 2:  
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
                		printf(" %f", (float)(*(data+offset[j][3]+y)));
                	printf("\n");

				}
				
				if(0 && flag == 1){
					FILE* relu_in = fopen("relu_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 24; l++)
							for(int m = 0; m < 24; m++)
								for(int n = 0; n < 20; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][5]+((l*24+m)*20+n)*img_num+p))); 
				}
				
				if(SCALE_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("scale_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], true);
				}
				
				if(SCALE_WEIGHT && SCALE_BIAS)
					scale_back(data+offset[j][2], wg+offset[j][8], data+offset[j][5], params[j], sparams[j]);
				else
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
				   	printf("scale_bias_after_backward:\n");
                	for(int y = 0; y < 20; y++) 
                		printf(" %f", (float)(*(data+offset[j][3]+y)));
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

				if(0 && j==5 && flag == 1){
					FILE* relu_in = fopen("fc2_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 100; l++)
							fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][5]+l*img_num+p)));
				}
				
				if(0 && j==6 && flag == 1){
					FILE* relu_in = fopen("softmax_indiff.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 10; l++)
							fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][5]+l*img_num+p)));
				}
				
				if(FULL_BACKWARD){
					if(j == 5)
						if(SHOW_QUANTIZE_RESULT)
							printf("fc1_outdiff_q : ");
					else
						if(SHOW_QUANTIZE_RESULT)
							printf("fc2_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j][1]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j][1]*img_num, &sparams[j][2], true);
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
				
				if(FULL_WEIGHT && FULL_BIAS)
			    	back_fc(data+offset[j][5], wg+offset[j][8], tmp, data+offset[j][2], params[j], sparams[j]);
				else
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
				
	       		if(0 && j==6 && flag == 1){
					FILE* relu_in = fopen("fc_w2.txt","w");
					for(int p = 0; p < 100; p++)
						for(int l = 0; l < 10; l++)
							fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][1]+l*100+p)));
				}
				
	       		if(0 && j==5 && flag == 1){
					FILE* relu_in = fopen("fc_w1.txt","w");
					for(int p = 0; p < 12; p++)
						for(int l = 0; l < 12; l++)
							for(int m = 0; m < 20; m++)
								for(int n = 0; n < 100; n++)	
									fprintf(relu_in, "%.6f\n", (float)(*(data+offset[j][1]+((p*12+l)*2000+n*20+m)))); 
				}	
			    break;
		}           
	}
}






