#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>

int main(){
    #define num_batch 1
	#define batch_size 2048
	#define BILLION 1000000000.0
	int print_flag = 0;

	int m = 27;
	int n = 27;
	int k = 128;
	int i, j;
	float a[batch_size * m * k];
	float b[batch_size * n * k];
	float c[batch_size * m * n];

	for(i = 0; i < batch_size * m * k; i++) {
		a[i] = (float)rand()/(float)(RAND_MAX) + 0.01;
	}

	for(i = 0; i < batch_size * n * k; i++) {
		b[i] = (float)rand()/(float)(RAND_MAX) + 0.01;
	}
	
	for(i = 0; i < batch_size * m * n; i++) {
		c[i] = 0.0;
	}

	/* [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
		dot
	   [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
	   1 * 1 + 2 * 6 + 3 * 11 + 4 * 16 = 110
	*/

	MKL_INT m_mkl[num_batch] = {m};
	MKL_INT n_mkl[num_batch] = {n};
	MKL_INT k_mkl[num_batch] = {k};

	MKL_INT lda[num_batch] = {batch_size * k};
	MKL_INT ldb[num_batch] = {batch_size * n};
	MKL_INT ldc[num_batch] = {batch_size * n};

	CBLAS_TRANSPOSE transA[num_batch] = { CblasNoTrans };
	CBLAS_TRANSPOSE transB[num_batch] = { CblasNoTrans };
	
	float alpha[num_batch] = {1.0};
	float beta[num_batch] = {0.0};
	
	const MKL_INT size_per_grp[num_batch] = {batch_size};
	
	const float *a_array[batch_size], *b_array[batch_size];
	float *c_array[batch_size];
	
	for (i = 0; i < batch_size; i++) {
		a_array[i] = a + i * k;
		b_array[i] = b + i * n;
		c_array[i] = c + i * n;
	}

	struct timespec start, end;
	int num_iter = 100000;

	void print_data(float data[], int r, int c, int ld) {
		if(print_flag) {
			int len = r * c;
			for(i = 0; i < len; i++) {
				printf("%f ", data[i]);
				if((i + 1) % ld == 0)
					printf("\n");
			}
			printf("\n");
		}
	}

	clock_gettime(CLOCK_REALTIME, &start);
        for (i = 0 ; i < num_iter; i++) { 
			cblas_sgemm_batch (CblasRowMajor, transA, transB,
				m_mkl, n_mkl, k_mkl, alpha,
				a_array, lda,
				b_array, ldb, beta,
				c_array, ldc,
				num_batch, size_per_grp);
		}
	clock_gettime(CLOCK_REALTIME, &end);
	print_data(a, m, batch_size * k, batch_size * k);
	print_data(b, n, batch_size * k, batch_size * n);
	print_data(c, m, batch_size * n, batch_size * n);

	double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
	printf("time = %f\n", (double)time_spent * 1000 / num_iter);
	
	return 0;
}
