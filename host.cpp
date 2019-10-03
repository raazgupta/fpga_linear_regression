#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include "xcl2.hpp"
#include <errno.h>   // for errno
#include <limits.h>  // for INT_MAX
#include "linear_regression.hpp"

#define MAXROWS 2459
#define MAXCOLS 2
#define ALPHA 0.1
#define THETA0 0
#define THETA1 0

///////////////////////////////////////////////////////////////////////////////

int load_file_to_memory(const char *filename, char **result) {
	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		*result = NULL;
		return -1; // -1 means file opening fail
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *) malloc(size + 1);
	if (size != fread(*result, sizeof(char), size, f)) {
		free(*result);
		return -2; // -2 means file reading fail
	}
	fclose(f);
	(*result)[size] = 0;
	return size;
}

void load_csv_to_memory(const char *filename, int *data, char columnNames[MAXCOLS][14]) {
	FILE* fp = fopen(filename,"r");
	int rowIndex = 0;
	char line[128];
	char* token = NULL;
	if (fp != NULL)
	{
		while (fgets( line, sizeof(line), fp) != NULL && rowIndex < MAXROWS)
		{
		  int colIndex = 0;
		  for (token = strtok( line, ","); token != NULL && colIndex < MAXCOLS; token = strtok(NULL, ","))
		  {
			  if(rowIndex == 0){
				  //printf("%s\n", token);
				  strcpy(columnNames[colIndex], token);
			  }
			  else {
				  //printf("%d\n",atoi(token));
				  data[(rowIndex-1)*MAXCOLS+colIndex] = atoi(token);
			  }
			colIndex++;
		  }
		  rowIndex++;
		}
		fclose(fp);
	 }
}

int main(int argc, char** argv) {

	/*
	if (argc != 2) {
		printf("format: host xclbin Int1 Int2");
		return EXIT_FAILURE;
	}
	*/

	const char *filename = "/home/centos/workspace/linear_regression/src/lin_reg_data_sample_3.csv";

	char columnNames[MAXCOLS][14];

	int *data = (int*) malloc(sizeof(int) * (MAXROWS-1) * MAXCOLS);

	load_csv_to_memory(filename, data, columnNames);

	for (int i = 0; i < MAXROWS; ++i)
	  {
	    for (int j = 0; j < MAXCOLS; ++j){
	    	if(i == 0){
	    		printf("%s ", columnNames[j]);
	    	}
	    	else {
	    		printf("%d ", data[(i-1)*MAXCOLS + j]);
	    	}
	    }
	    printf("\n");
	  }

	float alpha[1];
	float theta0[1];
	float theta1[1];
	float rsquared[1];
	alpha[0] = ALPHA;
	theta0[0] = THETA0;
	theta1[0] = THETA1;
	rsquared[0] = 0.0;

	//linear_regression(data, alpha, theta0, theta1, rsquared);




	// OPENCL HOST CODE AREA START
	// get_xil_devices() is a utility API which will find the xilinx
	// platforms and will return list of devices connected to Xilinx platform
	int err;
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];

	OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

	// find_binary_file() is a utility API which will search the xclbin file for
	// targeted mode (sw_emu/hw_emu/hw) and for targeted platforms.
	std::string binaryFile = xcl::find_binary_file(device_name,"linear_regression");

	// import_binary_file() ia a utility API which will load the binaryFile
	// and will return Binaries.
	cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
	devices.resize(1);
	OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
	OCL_CHECK(err, cl::Kernel krnl_linear_regression(program,"linear_regression", &err));

	// Allocate Buffer in Global Memory
	// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
	// Device-to-host communication
	OCL_CHECK(err, cl::Buffer input_data  (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * (MAXROWS-1) * MAXCOLS, data, &err));
	OCL_CHECK(err, cl::Buffer input_alpha   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			sizeof(float), alpha, &err));
	OCL_CHECK(err, cl::Buffer output_theta0 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
			sizeof(float), theta0, &err));
	OCL_CHECK(err, cl::Buffer output_theta1 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				sizeof(float), theta1, &err));
	OCL_CHECK(err, cl::Buffer output_rsquared (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				sizeof(float), rsquared, &err));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({input_data, input_alpha, output_theta0, output_theta1, output_rsquared},0)); //0 means from host

	OCL_CHECK(err, err = krnl_linear_regression.setArg(0, input_data));
	OCL_CHECK(err, err = krnl_linear_regression.setArg(1, input_alpha));
	OCL_CHECK(err, err = krnl_linear_regression.setArg(2, output_theta0));
	OCL_CHECK(err, err = krnl_linear_regression.setArg(3, output_theta1));
	OCL_CHECK(err, err = krnl_linear_regression.setArg(4, output_rsquared));

	// Launch the Kernel
	// For HLS kernels global and local size is always (1,1,1). So, it is recommended
	// to always use enqueueTask() for invoking HLS kernel
	OCL_CHECK(err, err = q.enqueueTask(krnl_linear_regression));

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({output_theta0, output_theta1, output_rsquared},CL_MIGRATE_MEM_OBJECT_HOST));

	//q.finish();
	OCL_CHECK(err, err = q.finish());

// OPENCL HOST CODE AREA END

	printf("theta0: %.6f \n", theta0[0]);
	printf("theta1: %.6f \n", theta1[0]);
	printf("R Squared: %.6f \n", rsquared[0]);

	return EXIT_SUCCESS;
}
