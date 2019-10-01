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

#define MAXROWS 2488
#define MAXCOLS 2

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

void load_csv_to_memory(const char *filename, int *data) {
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
			printf("%d\n",atoi(token));
			data[rowIndex*MAXCOLS+colIndex] = atoi(token);
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

	const char *filename = "/home/centos/workspace/linear_regression/src/lin_reg_data_sample.csv";

	int *data = (int*) malloc(sizeof(int) * MAXROWS * MAXCOLS);

	load_csv_to_memory(filename, data);

	for (int i = 0; i < MAXROWS; ++i)
	  {
	    for (int j = 0; j < MAXCOLS; ++j)
	      printf("%d ", data[i*MAXCOLS + j]);
	    printf("\n");
	  }


	/*
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
	std::string binaryFile = xcl::find_binary_file(device_name,"int_sum");

	// import_binary_file() ia a utility API which will load the binaryFile
	// and will return Binaries.
	cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
	devices.resize(1);
	OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
	OCL_CHECK(err, cl::Kernel krnl_int_add(program,"int_sum", &err));

	// Allocate Buffer in Global Memory
	// Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
	// Device-to-host communication
	OCL_CHECK(err, cl::Buffer input_a  (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int), a, &err));
	OCL_CHECK(err, cl::Buffer input_b   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			sizeof(int), b, &err));
	OCL_CHECK(err, cl::Buffer output_sum (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
			sizeof(int), sum, &err));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({input_a, input_b, output_sum},0)); //0 means from host

	OCL_CHECK(err, err = krnl_int_add.setArg(0, input_a));
	OCL_CHECK(err, err = krnl_int_add.setArg(1, input_b));
	OCL_CHECK(err, err = krnl_int_add.setArg(2, output_sum));

	// Launch the Kernel
	// For HLS kernels global and local size is always (1,1,1). So, it is recommended
	// to always use enqueueTask() for invoking HLS kernel
	OCL_CHECK(err, err = q.enqueueTask(krnl_int_add));

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({output_sum},CL_MIGRATE_MEM_OBJECT_HOST));

	//q.finish();
	OCL_CHECK(err, err = q.finish());

// OPENCL HOST CODE AREA END
*/
	return EXIT_SUCCESS;
}
