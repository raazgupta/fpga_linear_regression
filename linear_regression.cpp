#include <math.h>

#define MAXROWS 2459
#define MAXCOLS 2
#define DATA_SIZE (MAXROWS-1)*MAXCOLS

extern "C" {

float calc_cost(int *data, float *theta0, float *theta1){
	float m = MAXROWS - 1;
	float sum_cost = 0.0;
	int y = 0;
	int x = 0;
	float h = 0.0;

	calc_cost_for: for(int i=0; i<m; i++){
#pragma HLS PIPELINE
		y = data[i*MAXCOLS];
		x = data[i*MAXCOLS + 1];
		h = theta0[0] + theta1[0]*x;
		sum_cost = sum_cost + pow(h-y,2);
	}

	float cost = 1 / (2*m) * sum_cost;
	return cost;
}

void calc_rsquared(int *data, float *theta0, float *theta1, float *rsquared){
	float m = MAXROWS - 1;
	float sse = 0.0;
	float ssw = 0.0;
	float sum_y = 0.0;
	float mean_y = 0.0;
	int y = 0;
	int x = 0;
	float h = 0.0;

	rsquared_for: for(int i=0; i<m; i++){
#pragma HLS PIPELINE
		y = data[i*MAXCOLS];
		x = data[i*MAXCOLS + 1];
		h = theta0[0] + theta1[0]*x;

		sse = sse + pow((y - h),2);
		sum_y = sum_y + y;
	}

	mean_y = sum_y / m;
	ssw_for: for(int i=0; i<m; i++){
#pragma HLS PIPELINE
		y = data[i*MAXCOLS];
		ssw = ssw + pow((y-mean_y),2);
	}

	rsquared[0] = 1 - (sse/ssw);
}

void linear_regression(int data[DATA_SIZE], float *alpha, float *theta0, float *theta1, float *rsquared){

#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=alpha offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=theta0 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=theta1 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=rsquared offset=slave bundle=gmem1

#pragma HLS INTERFACE s_axilite port=data bundle=control
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=theta0 bundle=control
#pragma HLS INTERFACE s_axilite port=theta1 bundle=control
#pragma HLS INTERFACE s_axilite port=rsquared bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=data cyclic factor=2

	float m = MAXROWS - 1;
	float sum_theta0 = 0.0;
	float sum_theta1 = 0.0;
	float current_cost = 0.0;
	float previous_cost = 0.0;
	float previous_theta0 = theta0[0];
	float previous_theta1 = theta1[0];
	int step = 0;
	int y = 0;
	int x = 0;
	float h = 0.0;

	previous_cost = calc_cost(data, theta0, theta1);

	do{
		sum_theta0 = 0.0;
		sum_theta1 = 0.0;

		if(step > 0){
			previous_cost = current_cost;
		}

		linear_regression_for: for(int i=0; i<m; i++){
#pragma HLS PIPELINE

			y = data[i*MAXCOLS];
			x = data[i*MAXCOLS + 1];
			h = theta0[0] + theta1[0]*x;

			sum_theta0 = sum_theta0 + (h-y);
			sum_theta1 = sum_theta1 + (h-y)*x;

		}
		previous_theta0 = theta0[0];
		previous_theta1 = theta1[0];
		theta0[0] = theta0[0] - alpha[0] / m * sum_theta0;
		theta1[0] = theta1[0] - alpha[0] / m * sum_theta1;

		current_cost = calc_cost(data, theta0, theta1);

		step++;
	}while(current_cost < previous_cost);

	theta0[0] = previous_theta0;
	theta1[0] = previous_theta1;

	calc_rsquared(data, theta0, theta1, rsquared);

}
}
