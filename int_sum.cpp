#include <stdio.h>
#include <string.h>
#include <ap_int.h>



extern "C" {

	void int_sum(int *a, int *b, int *sum)
	{
	#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2


	#pragma HLS INTERFACE s_axilite port=a bundle=control
	#pragma HLS INTERFACE s_axilite port=b bundle=control
	#pragma HLS INTERFACE s_axilite port=sum bundle=control

	#pragma HLS INTERFACE s_axilite port=return bundle=control

		sum[0] = a[0] + b[0];

		return;
	}
}
