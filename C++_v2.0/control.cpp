#include "conv.h"

#define layer_num 94

void training(float *data, float *wg, float *fp, float *tmp, int *setting, int *sp, int rate, int epoch, int decay, int flag)
{
    int params[layer_num][16];
    int offset[layer_num][10];
    int connect[layer_num][4];
    
    
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

				if(0 && flag == 1 && j == 0){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 3; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][0]+(l*32+m)*3*img_num+img_num*n+p))); 
					printf("F!!!");
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
				
				if (CONV_WEIGHT)
    	        	conv(data+offset[j][0], wg+offset[j][8], tmp, data+offset[j][4], params[j], sparams[j]);
				else
					conv(data+offset[j][0], data+offset[j][1], tmp, data+offset[j][4], params[j], sparams[j]);
               	break;

       		case 1: 
				
       			if(0 && flag == 1 && j == 14){
					FILE* batch_in_q = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)	
									fprintf(batch_in_q, "%.6f\n", (float)(*(data+offset[j][0]+((l*32+m)*16+n)*img_num+p))); 
					printf("F!!!");
				}
				
			   	if(BATCH_FORWARD){
			   		if(SHOW_QUANTIZE_RESULT)
						printf("batch_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}
              
    	        batch(data+offset[j][0], wg+offset[j][1], data+offset[j][4], params[j], sparams[j]);
    	        
    	        break;

       		case 2:  
				
				if(0 && flag == 1 && j == 2){
					FILE* scale_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)	
									fprintf(scale_in, "%.6f\n", (float)(*(data+offset[j][0]+((l*32+m)*16+n)*img_num+p))); 
					printf("F!!!");
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

               	if(0 && flag == 1 && j == 7){
					FILE* batch_in_q = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)	
									fprintf(batch_in_q, "%.16f\n", (float)(*(data+offset[j][0]+((l*32+m)*16+n)*img_num+p))); 
					printf("F!!!");
				}
				
               	relu(data+offset[j][0], data+offset[j][4], params[j], sparams[j]);

               	
                break;

       		case 4: 
			   
				if(0 && flag == 1 && j == 91){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 8; l++)
							for(int m = 0; m < 8; m++)
								for(int n = 0; n < 64; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][0]+(l*8+m)*64*img_num+img_num*n+p))); 
					printf("F!!!");
				}  

               	pool(data+offset[j][0], data+offset[j][1], data+offset[j][4], params[j], sparams[j]);
               	
            	break;

       		case 5: 
       	
				eltwise(data+offset[j][0], data+offset[j][7], data+offset[j][4], params[j], sparams[j]);
				
			   	break;

       		case 6:  

				if(0 && flag == 1 && j == 92){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 1; l++)
							for(int m = 0; m < 1; m++)
								for(int n = 0; n < 64; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][0]+(l*1+m)*64*img_num+img_num*n+p))); 
					printf("F!!!");
				}
				
			   	if(FULL_FORWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("fc_input_q : ");
					quantize_backward(data+offset[j][0], params[j][0]*params[j][4]*params[j][5]*img_num, &sparams[j][0], false);
				}

			   	if(FULL_WEIGHT){
					if(SHOW_QUANTIZE_RESULT)
						printf("fc_weight_q : ");
					quantize_forward(data+offset[j][1], wg+offset[j][8], params[j][0]*params[j][1]*params[j][4]*params[j][5], &sparams[j][4], false);
				}

			   	if(FULL_BIAS){
					if(SHOW_QUANTIZE_RESULT)
						printf("fc_bias_q : ");
					quantize_forward(data+offset[j][3], wg+offset[j][9], params[j][1], &sparams[j][5], false);
				}

				if(FULL_WEIGHT && FULL_BIAS)
    	        	fc(data+offset[j][0], wg+offset[j][8], wg+offset[j][9], data+offset[j][4], params[j], sparams[j]);
    	        else
    	        	fc(data+offset[j][0], data+offset[j][1], data+offset[j][3], data+offset[j][4], params[j], sparams[j]);
    	        break;
    	        
       		case 7:  
       		
    	        if(0 && flag == 1 && j == 93){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 1; p++)
						for(int n = 0; n < 10; n++)
							//fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][0]+img_num*n+p)));
							printf("%.6f ", (float)(*(data+offset[j][0]+img_num*n+p)));
					printf(" \n"); 	
				}	

              	if(SHOW_SOFTMAX_IN) 
              	{  
					printf("softmax_input:\n");
                	for(int i = 0; i < 10; i++) 
                		printf(" %f", (float)(*(data+offset[j][0]+i*img_num)));
                	printf("\n");  
				}
									
    	        softmax(data+offset[j][0], data+offset[j][4], params[j], sparams[j], data+offset[j][2], (float*)(wg+offset[j][1]));
    	        
    	        if(0 && flag == 1 && j == 93){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int n = 0; n < 10; n++)
							fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+img_num*n+p)));
					printf("F!!!"); 
				}    	        
   
       	}
 	}
     
	for(int j = layer_num-2; j>=0; j--)
	{
		switch(params[j][10])
		{
			case 0: 
							
				// Important: Change 24 
				if(CONV_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("conv_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j+1][4]*params[j+1][5]*params[j][1]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j+1][4]*params[j+1][5]*params[j][1]*img_num, &sparams[j][2], true);
				}
				
				if(CONV_WEIGHT)
			    	conv_back(data + offset[j][5], wg+offset[j][8], data+offset[j][2], tmp, params[j], sparams[j]);
				else
					conv_back(data + offset[j][5], data+offset[j][1], data+offset[j][2], tmp, params[j], sparams[j]);
					
		        weight_back(data + offset[j][5], data+offset[j][1], data+offset[j][0], wg+offset[j][6], params[j], sparams[j], rate);

				if(0 && flag == 1 && j == 0){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 3; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+(l*32+m)*3*img_num+img_num*n+p))); 
					printf("F!!!");
				}
				
				if(0 && flag == 1 && j == 35){
                	FILE* conv_in_w = fopen("conv_in.txt","w");
                	for(int p = 0; p < 3; p++)
                		for(int l = 0; l < 3; l++)
                			for(int m = 0; m < 32; m++)
                				for(int n = 0; n < 32; n++)
									fprintf(conv_in_w, "%.6f\n", (float)(*(data+offset[j][1] + (p*3+l)+m*9+n*9*32)));
					printf("F!!!"); 
				}
				
				
                break;

			case 1: 				
				if(BATCH_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("batch_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j][1]*params[j][4]*params[j][5]*img_num, &sparams[j][2], true);
				}

			    batch_back(data+ offset[j][4], data+ offset[j][2], wg+offset[j][1], data+ offset[j][5], params[j], sparams[j]);
			    
				if(0 && flag == 1 && j == 5){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+(l*32+m)*16*img_num+img_num*n+p))); 
					printf("F!!!");
				}
                break;

			case 2: 
				
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
				
				if(0 && flag == 1 && j == 6){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+(l*32+m)*16*img_num+img_num*n+p))); 
					printf("F!!!");
				}
				
				if(0 && flag == 1 && j == 40){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 32; p++)
						fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][3]+p))); 
					printf("F!!!");
				}
                break;
                
			case 3: 
				if(0 && flag == 1 && j == 3){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][5]+(l*32+m)*16*img_num+img_num*n+p))
										+ (float)(*(data+offset[j][6]+(l*32+m)*16*img_num+img_num*n+p))); 
					printf("F!!!");
				}
											
			    relu_back(data+offset[j][0], data+offset[j][5], data+offset[j][6], data+offset[j][2], params[j], sparams[j]);
							    
				if(0 && flag == 1 && j == 3){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 32; l++)
							for(int m = 0; m < 32; m++)
								for(int n = 0; n < 16; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+(l*32+m)*16*img_num+img_num*n+p))); 
					printf("F!!!");
				}
		        break;

			case 4: 

			    back_pool(data+offset[j][2], data+offset[j][1], data+offset[j][5], data+offset[j][6], params[j], sparams[j]);
			    
				if(0 && flag == 1 && j == 91){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int l = 0; l < 8; l++)
							for(int m = 0; m < 8; m++)
								for(int n = 0; n < 64; n++)
									fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+(l*8+m)*64*img_num+img_num*n+p))); 
					printf("F!!!");
				}
				
			    break;

			case 5:
				
				eltwise_back(data+offset[j][5], data+offset[j][2], params[j], sparams[j]); 
				
				break;

			case 6: 	
				
				if(FULL_BACKWARD){
					if(SHOW_QUANTIZE_RESULT)
						printf("fc_outdiff_q : ");
					if(TEST_MODE)
						quantize_backward(data+offset[j][5], params[j][1]*img_num, &sparams[j][2], false);
					else
						quantize_backward(data+offset[j][5], params[j][1]*img_num, &sparams[j][2], true);
				}
				
				if(FULL_WEIGHT && FULL_BIAS)
			    	back_fc(data+offset[j][5], wg+offset[j][8], tmp, data+offset[j][2], params[j], sparams[j]);
				else
					back_fc(data+offset[j][5], data+offset[j][1], tmp, data+offset[j][2], params[j], sparams[j]);
			    fc_weight(data+offset[j][5], data+offset[j][1], data+offset[j][0], wg+offset[j][6], params[j], sparams[j], rate);
			    bias_back_fc(data+offset[j][5], data+offset[j][3], wg+offset[j][7], params[j], sparams[j], rate);
				
				if(0 && flag == 1 && j == 92){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 16; p++)
						for(int n = 0; n < 64; n++)
							fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][2]+img_num*n+p)));
					printf("F!!!"); 
				} 	
				
				if(0 && flag == 1 && j == 92){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int p = 0; p < 64; p++)
						for(int n = 0; n < 10; n++)
							fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][1]+n*64+p)));
					printf("F!!!");
				}

				if(0 && flag == 1 && j == 92){
					FILE* conv_in = fopen("conv_in.txt","w");
					for(int n = 0; n < 10; n++)
						fprintf(conv_in, "%.6f\n", (float)(*(data+offset[j][3]+n)));
					printf("F!!!");
				}
			    break;
		}           
	}
}







