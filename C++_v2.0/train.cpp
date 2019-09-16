#include <cmath>
#include <random>
#include <fstream>
#include "conv.h"
#include "table.h"

#define PI 3.14159265
#define img_num 16 
#define layer_num 94

#include <cstdlib>
#include <cmath>
#include <limits>

double normdis(double mu, double sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846; 

    static double z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	{
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	}
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}


int main(int argc, char const *argv[])
{   
 	int flag_success = 1;
    int epoch = 100;
    int decay = 2000;
    float lr = 0.01;
    int rates[layer_num];
	int count = 0;
	int Flag = 0 ;
	
    int all_num = 0;
    int true_num = 0;
    int aa, bb;
 
    for(int i = 0; i<epoch; i++)
    {
       float lrt =  0.005 * cos(i*PI/2/100);
       rates[i] = 1/lrt;
    }

    FILE* outloss;
    outloss = fopen("outputloss.txt","w");
    FILE* outs;
    outs = fopen("outputs.txt","w");
    FILE* _data;
    _data = fopen("data.txt","r");

    FILE* _lable;
    FILE* weight1;
    FILE* weight2;
    FILE* weight3;

    _lable = fopen("lable.txt","r");
	
	float fp[2];
    float * data =  (float*)malloc(113774244*4);
    float * wg = (float*)malloc(547012*4);
    float * label = (float*)&data[14348538];
    float * res = (float*)&wg[545732];    
	float * tmp = (float*)malloc(1048576*4);
    int params[layer_num][16];
    int offset[layer_num][10];
    int connect[layer_num][4];
    int sparams[layer_num][8];
	
 	for(int i = 0; i<layer_num; i++)
	{
		memcpy(params[i], (int*)ins+30*i, 16*4);
		memcpy(offset[i], (int*)ins+30*i+16, 10*4);
		memcpy(connect[i], (int*)ins+30*i+26, 4*4);
		sparams[i][0] = 5;
		if(TEST_MODE)
			sparams[i][2] = 15;
		else	
			sparams[i][2] = 15;
		sparams[i][4] = 5;
		sparams[i][5] = 5;
	}
    float sample;
    int w_num, innode;

    for(int i = 0; i<layer_num; i++)
    {
      	printf("here initilize weight at %d and %d\n", i, ins[i][10]);
      	switch(ins[i][10])
      	{
       		case 0:				
				w_num =  params[i][0]*params[i][1]*params[i][8]*params[i][8];
				innode = params[i][0]*params[i][8]*params[i][8]; 

     			weight1 = fopen("weight1.txt", "r");
           		for(int j = 0; j < w_num; j++)
           		{
             		sample = normdis(0,0.1);
             
					float fVal = 0;
					fscanf(weight1, "%f", &fVal);
					int m1 = j%params[i][1];
					int n1 = j/params[i][1];
					int m2 = n1%params[i][0];
					int n2 = n1/params[i][0];

					if(TEST_MODE)
             			data[offset[i][1] + n2 + m2*params[i][8]*params[i][8] + m1*innode] = fVal;
					else
             			data[offset[i][1]+j] = sample;

           		}
           		break;

       		case 2:
				w_num =  params[i][1];
				for(int j = 0; j < w_num; j++)
				{	
					if(TEST_MODE){
					    data[offset[i][1]+j] = 0.35;  
					    data[offset[i][3]+j] = 0.05;
					}
					else{
						data[offset[i][1]+j] = 1;  
					    data[offset[i][3]+j] = 0;	
					}
				}
				break;

       		case 6:
				w_num =  params[i][0]*params[i][1]*params[i][4]*params[i][4];
				innode = params[i][0]*params[i][4]*params[i][4]; 
				if(i==5)
					weight2 = fopen("weight2.txt", "r");
				else
					weight3 = fopen("weight3.txt", "r");

				float fVal = 0;
				for(int j = 0; j < w_num; j++)
				{ 
					if(i==5){
						sample = normdis(0,0.1);			
						fscanf(weight2,"%f",&fVal);
						int m1 = j%100;
						int n1 = j/100;
						int m2 = n1%20; 
						int n2 = n1/20;

						if(TEST_MODE)
							data[offset[i][1]+n2*2000+m1*20+m2] = fVal;
						else
							data[offset[i][1]+j] = sample;
					}
					else{
						sample = normdis(0,0.1);
						int m1 = j%10;
						int n1 = j/10;
						fscanf(weight3,"%f",&fVal);

						if(TEST_MODE)
							data[offset[i][1]+m1*64+n1] = fVal;
						else
							data[offset[i][1]+j] = sample;
		 			}
			 	}

				for(int j = 0;j < params[i][1]; j++) 
					if(TEST_MODE) 
						data[offset[i][3]+j] = 0.01;
					else
						data[offset[i][3]+j] = 0.1;
				
				printf("\nfull: finish \n");
				break;     
    	}
	}    
	
		
    std::ifstream file;
   	for(int ep = 0; ep<10; ep++)
  	{ 
		int idx = 0;
		all_num = 0;
		true_num = 0;
		
		while(idx <(50000/img_num)*img_num)
		{  
    		if(idx == 0) 
    			file.open("data_batch_1.bin", std::ios::binary);
    		else if(idx == 10000){
				file.close();
     			file.open("data_batch_2.bin", std::ios::binary);
    		}else if(idx == 20000){
				file.close();
     			file.open("data_batch_3.bin", std::ios::binary);
    		}else if(idx == 30000){
				file.close();
     			file.open("data_batch_4.bin", std::ios::binary);
    		}else if(idx == 40000){
				file.close();
     			file.open("data_batch_5.bin", std::ios::binary);}
     			
    		if(file.is_open())
    		{
				int im = idx%img_num;
				//if(idx<=15) {
				unsigned char labelno;
				file.read((char*)&labelno, 1);
				for(int i = 0; i<10 ;i++)
				{
					if(int(labelno) == i)
						label[i*img_num+im] = 1;
					else
						label[i*img_num+im] = 0;
					//printf("%f\n", label[i*img_num+im]);
     			}     
						 				
     			for(int i = 0; i < 3; i++)
      				for(int y = 0; y < 32; y++)
       					for(int x = 0 ; x < 32; x++)
         				{
           					unsigned char tmp;
           					file.read((char*)&tmp, 1); 
           					data[(y*32+x)*3*img_num+i*img_num+im] = (float(tmp) - 128)/128; 
							//printf("%f ", data[(y*32+x)*3*img_num+i*img_num+im]);    
         				} 
         					//}	
     			idx++;
     			
     			if(idx%img_num == 0)
     			{
					if(TEST_MODE){
						int last = 1;
						if(flag_success == 1 && idx%16 == 0){
							for (int e = 0; e<last; e++){
								_data = fopen("data.txt","r");
								for (int mm = 0; mm<16; mm++)
									for (int nn = 0; nn<32; nn++)
										for (int ee = 0; ee<32; ee++)
											for(int ii = 0; ii<3; ii++){
												float fVal = 0;
												fscanf(_data, "%f", &fVal);
												data[(nn*32+ee)*3*img_num+ii*img_num+mm] = fVal;						
											}
								if(e == last - 1)
									training(data, wg, fp, tmp, (int*)ins, (int*)sparams, rates[ep], epoch, decay, 1);
								else
									training(data, wg, fp, tmp, (int*)ins, (int*)sparams, rates[ep], epoch, decay, 0);
							}
							flag_success = 2;
							printf("Finished!");
						}
					}
					else
						training(data, wg, fp, tmp, (int*)ins, (int*)sparams, rates[ep], epoch, decay, 1);
     				float loss = 0;     

					for(int i = 0; i<img_num;i++)
					{
						for(int j = 0; j < 10; j++) 
							{ 
								if(label[j*img_num + i] == 1)
								loss += -log(res[j*img_num+i]);
							}
					}	  

					if(PRINT_LOSS)
						printf("************************epoch:%d batch:%d loss:%f\n", ep, idx/img_num, loss);
 
					//fprintf(outloss,"\n"); 
					//fprintf(outloss,"epoch:%d batch:%d loss:%f\n", ep, idx/img_num, loss); 
      
					
					for(int i = 0; i<img_num; i++)
					{
						for(int j = 0; j < 10; j++) 
						{ 
							if(label[j*img_num+i] == 1){
								aa = j;
								if(SHOW_PAIRS)
									printf("%d - ", j);
							}
						}

						float maxmax = res[0*img_num+i];
						int goal = 0;

						for(int j = 0; j < 10; j++) 
						{ 
							if(res[j*img_num+i] > maxmax)
							{
								maxmax = res[j*img_num + i];
								goal = j;
							}
						}

						bb = goal;
						if(SHOW_PAIRS)
							printf("%d\n", goal);

						if(ep >= 0){
							all_num += 1;
							if(aa == bb)
								true_num += 1;
						}
						
					}
					
					if(PRINT_ACCURACY){
						if(ep >= 0)
							printf("Accuracy : %f\n", 1.0*true_num/all_num);
					}
      
     			}
     
    		}
    		else
      			printf("error: no file open\n");     
   		}

	file.close();
   	}

    return 0;
}

