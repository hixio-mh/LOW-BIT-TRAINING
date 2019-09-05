#include <cmath>
#include <random>
#include <fstream>
#include "conv.h"
#include "table.h"

#define PI 3.14159265
#define img_num 16 
#define layer_num 8

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
    _data = fopen("data.txt","w");

    FILE* _lable;
    FILE* weight1;
    FILE* weight2;
    FILE* weight3;

    _lable = fopen("lable.txt","w");

	float fp[2];
    float * data =  (float*)malloc(13422194*4);
    float * wg = (float*)malloc(580600*4);
    float * label = (float*)&data[7039602];
    float * res = (float*)&wg[579320];    
    float * temp_input = (float*)malloc(184320*4);  
    float * temp_weight = (float*)malloc(288000*4);  
    float * temp_bias = (float*)malloc(100*4);  

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
		if(i == 0)
			sparams[i][0] = 8;
		sparams[i][2] = 12;
		sparams[i][4] = 8;
		sparams[i][5] = 10;
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
					int m1 = j%20;
					int n1 = j/20;

					if(TEST_MODE)
             			data[offset[i][1]+m1*25+n1] = fVal;
					else
             			data[offset[i][1]+j] = sample;

           		}
           		break;

       		case 2:
				w_num =  params[i][1];
				for(int j = 0; j < w_num; j++)
				{
					data[offset[i][1]+j] = 1;  
					data[offset[i][3]+j] = 0;
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
							data[offset[i][1]+j] = 0.2*sample;
					}
					else{
						sample = normdis(0,0.1);
						int m1 = j%10;
						int n1 = j/10;
						fscanf(weight3,"%f",&fVal);

						if(TEST_MODE)
							data[offset[i][1]+m1*100+n1] = fVal;
						else
							data[offset[i][1]+j] = sample;
		 			}
			 	}

				for(int j = 0;j < params[i][1]; j++) 
					data[offset[i][3]+j] = 0.1;
				
				printf("\nfull: finish \n");
				break;     
    	}
	}    
	
		
    std::ifstream datafile;
    std::ifstream labelfile;
   	for(int ep = 0; ep<10; ep++)
  	{ 
		int idx = 0;
		int header = 0;
		all_num = 0;
		true_num = 0;
		datafile.open("train-images.idx3-ubyte", std::ios::binary);
		labelfile.open("train-labels.idx1-ubyte", std::ios::binary);
    
		if(datafile.is_open() && labelfile.is_open())
		{
			printf("here `````````````````\n");
			datafile.read((char*)&header, sizeof(int));
			datafile.read((char*)&header, sizeof(int));
			printf("data file %d ",header);
			datafile.read((char*)&header, sizeof(int));
			printf("%d ",header);
			datafile.read((char*)&header, sizeof(int));
			printf("%d\n",header);
			labelfile.read((char*)&header, sizeof(int));
			labelfile.read((char*)&header, sizeof(int));
			printf("label file %d\n",header);
  		}

		while(idx <(60000/img_num)*img_num)
		{  
			if(datafile.is_open() && labelfile.is_open())
			{
				int im = idx%img_num;
				unsigned char labelno;
				labelfile.read((char*)&labelno, 1);
				for(int i = 0; i<10 ;i++)
				{
					if(int(labelno) == i)
						label[i*img_num+im] = 1;
					else
						label[i*img_num+im] = 0;
     			}     

				for(int y = 0; y<28; y++)
					for(int x = 0; x<28; x++)
					{
						unsigned char tmp;
						datafile.read((char*)&tmp, 1); 
           				data[(y*28+x)*img_num+im] = (float(tmp) - 128)/128;   
         			}
     			idx++;
     			if(idx%img_num == 0)
     			{
					if(TEST_MODE){
						if(flag_success == 1 && idx%16 == 0){
							for (int e = 0; e<1; e++)
								training(data, wg, fp, (int*)ins, (int*)sparams, rates[ep], epoch, decay, temp_input, temp_weight, temp_bias);
							flag_success = 2;
						}
					}
					else
						training(data, wg, fp, (int*)ins, (int*)sparams, rates[ep], epoch, decay, temp_input, temp_weight, temp_bias);

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

					if(SHOW_SOFTMAX_OUT){			
						//FILE* softmax_out = fopen("softmax_out.txt","w");	
						for(int i = 0; i<img_num; i++)
						{
							loss = 0;
							for(int j = 0; j < 10; j++) 
							{ 
								if(label[j*img_num+i] == 1)
								{
									loss += -log(res[j*img_num+i]);
									//fprintf(softmax_out, "%.6f\n", -log(res[j*img_num+i]));
									printf("%d : %f\n", i, -log(res[j*img_num+i])); 
								}
							}
						}
					}
      
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

	datafile.close();
	labelfile.close();

   	}

    return 0;
}

