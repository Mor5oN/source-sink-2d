#ifndef SPATIALTISSUE_H
#define SPATIALTISSUE_H
#include <fstream>
#include <iostream>
#include <float.h>
#include <math.h>
#include <vector>
#include <algorithm>
using namespace std;
#ifndef PCG_THREADS_PER_BLOCK
#define PCG_THREADS_PER_BLOCK 256
#endif
#define BLOCK_SIZE_X PCG_THREADS_PER_BLOCK
#define BLOCK_SIZE_Y 1
#define pi 3.141592653589793
#define R 8314.0
#define Frdy 96485.0
#define Temp 310.0
#define FoRT (Frdy / R / Temp)

#define f_k1 0.0
#ifndef JUNCTION_AREA_FRACTION
#define JUNCTION_AREA_FRACTION 0.5
#endif
#ifndef MYO_ROWS
#define MYO_ROWS 1
#endif
#ifndef MYO_COLS
#define MYO_COLS number_of_myocyte
#endif
#ifndef PCG_MAX_ITER
#define PCG_MAX_ITER 80
#endif
#ifndef PCG_TOL
#define PCG_TOL 1e-12
#endif
#ifndef PCG_CHECK_INTERVAL
#define PCG_CHECK_INTERVAL 4
#endif
__host__ __device__ inline bool finite_value(double x)
{
	return (x == x) && (fabs(x) <= DBL_MAX);
}
#include "myocyte.cuh"
#include "Cell_Grid.cuh"
template <unsigned int NMYOS>
class SpatialTissue
{
public:
	enum { MAX_CLEFTS = Cell_Grid<NMYOS>::MAX_CLEFTS };
	dim3 dimBlock_myo, dimGrid_myo, dimBlock_grid, dimGrid_grid;
	dim3 dimBlock_cleft, dimGrid_cleft, dimBlock_matrix, dimGrid_matrix;
	cudaStream_t MyoStream, GridStream;
	myocyte<NMYOS>* rabbit_myocyte;
	myocyte<NMYOS>* d_rabbit_myocyte;
	Cell_Grid<NMYOS>* cable_grid;
	Cell_Grid<NMYOS>* d_cable_grid;
	double* d_output_voltage;
	int* d_pcg_status;
	double radius;
	double length;
	double A_m;
	double A_myo_j;
	double A_myo_j_vertical;
	double C_m;
	double V_Cell[NMYOS];
	double A_cell_lat[NMYOS];
	double A_cell_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double C_cell_lat[NMYOS];
	double C_cell_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double NUMofCells_in_grid[NMYOS];
	double NUMofCell_connected[MAX_CELL_NEIGHBORS][NMYOS];
	int cell_num_neighbors[NMYOS];
	double ryo_f_Ina_lat[NMYOS];
	double ryo_f_k1_lat[NMYOS];
	double ryo_f_Ina_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double ryo_f_k1_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double vertical_junc_radius;
	double vertical_Na_share;
	double vertical_K_share;
	bool vertical_gap_only_mode;
	int size_of_para_matrix;
	double f_Ina_cleft;
	double f_Ina_lateral;
	double stagger_lambda;
	__host__ __device__ inline double get_horizontal_junction_fraction(int cell_a, int cell_b) const
	{
		int row_a = cell_a / MYO_COLS;
		int row_b = cell_b / MYO_COLS;
		return (row_a == row_b) ? (1.0 - stagger_lambda) : stagger_lambda;
	}
	SpatialTissue(double vertical_junc_radius_input = 0.0,
				  double vertical_Na_share_input = 0.0,
				  double vertical_K_share_input = 0.0,
				  bool vertical_gap_only_mode_input = false,
				  double stagger_lambda_input = JUNCTION_AREA_FRACTION);
	~SpatialTissue();
	void arrange_cells_surface_and_Capacitance(double f_Ina_cleft_input, double f_Ina_lateral_input);
	void arrange_cleft_parameter(double *g_cleft_control,
								  double *horizontal_width_of_cleft_myo,
								  double *horizontal_G_gap_myo,
								  double *vertical_width_of_cleft_myo,
								  double *vertical_G_gap_myo);
	void deliver_parameter_to_class();
	void cpu_copy_gpu();
	void build_sparse_matrix();
	void copy_voltage_to_host();
	void step(double dt, double &Istim, double in_time, int stimulus_id);
	void build_staggered_topology();
	void add_cleft_connection(int cell_a, int cell_b);
	void add_direct_gap_connection(int cell_a, int cell_b);
	void arrange_center_stimulus_region(int num_stimulated_cells, int shape_type);
};

template <unsigned int NMYOS>
__global__ void Compute_Myocyte_ODE(double dt, myocyte<NMYOS>* rabbit_myocyte, Cell_Grid<NMYOS>* cable_grid, double Istim, double in_time, int stimulus_id);
template <unsigned int NMYOS>
__global__ void Compute_Cleft_ODE(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Arrange_Linear_Constant(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Prepare_PCG(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Finalize_Initial_PCG(Cell_Grid<NMYOS>* cable_grid, int num_blocks);
template <unsigned int NMYOS>
__global__ void PCG_MatVec_Direction_And_Dot(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void PCG_Finalize_Direction_Dot(Cell_Grid<NMYOS>* cable_grid, int num_blocks);
template <unsigned int NMYOS>
__global__ void PCG_Update_X_R_Z_And_Dot(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void PCG_Finalize_RZ_And_Check(Cell_Grid<NMYOS>* cable_grid, int num_blocks);
template <unsigned int NMYOS>
__global__ void PCG_Update_Direction(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Assignment_myocyte_dvdt(myocyte<NMYOS>* rabbit_myocyte, Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Assignment_Cleft_dvdt(Cell_Grid<NMYOS>* cable_grid);
template <unsigned int NMYOS>
__global__ void Copy_Voltage_For_Output(myocyte<NMYOS>* rabbit_myocyte, double* output_voltage);
template <unsigned int NMYOS>
__global__ void Copy_PCG_Status(Cell_Grid<NMYOS>* cable_grid, int* pcg_status);

template <unsigned int NMYOS>
__global__ void Myocyte_Update(double dt, myocyte<NMYOS>* rabbit_myocyte)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= NMYOS)
		return;
	rabbit_myocyte -> rabbit_update(id, dt);
}

template <unsigned int NMYOS>
__global__ void Cleft_update(double dt, Cell_Grid<NMYOS>* cable_grid)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= cable_grid -> num_clefts)
		return;
	cable_grid -> compute_dion_dt(idx);
	cable_grid -> cleft_update(id, dt);
}
#include "SpatialTissue.cu"
#endif
