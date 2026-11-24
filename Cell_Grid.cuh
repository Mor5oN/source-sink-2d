#ifndef CELL_GRID_H
#define CELL_GRID_H
#ifndef MAX_CELL_NEIGHBORS
#define MAX_CELL_NEIGHBORS 6
#endif
#ifndef PCG_THREADS_PER_BLOCK
#define PCG_THREADS_PER_BLOCK 256
#endif
template <unsigned int NMYOS>
class Cell_Grid
{
public:
	enum { MAX_CLEFTS = (NMYOS * MAX_CELL_NEIGHBORS) / 2 };
	enum { MAX_MATRIX_SIZE = NMYOS + MAX_CLEFTS };
	enum { MAX_MATRIX_NNZ = NMYOS * (MAX_CELL_NEIGHBORS + 1) + MAX_CLEFTS * 3 };
	enum { MAX_PCG_BLOCKS = (MAX_MATRIX_SIZE + PCG_THREADS_PER_BLOCK - 1) / PCG_THREADS_PER_BLOCK };
	double V_Cell[NMYOS];
	double Cell_lat_current[NMYOS];
	double Cell_junc_current[MAX_CELL_NEIGHBORS][NMYOS];
	double Cell_Na_junc_current[MAX_CELL_NEIGHBORS][NMYOS];
	double Cell_K1_junc_current[MAX_CELL_NEIGHBORS][NMYOS];
	double NUMofCells_in_grid[NMYOS];
	double NUMofCell_connected[MAX_CELL_NEIGHBORS][NMYOS];
	int stim_mask_first[NMYOS];
	int cell_num_neighbors[NMYOS];
	int cell_neighbor_cell[MAX_CELL_NEIGHBORS][NMYOS];
	int cell_neighbor_cleft[MAX_CELL_NEIGHBORS][NMYOS];
	int cell_num_direct_gaps[NMYOS];
	int cell_direct_gap_neighbor[MAX_CELL_NEIGHBORS][NMYOS];
	double cell_direct_gap_G[MAX_CELL_NEIGHBORS][NMYOS];

	int num_clefts;
	int cleft_cell_a[MAX_CLEFTS];
	int cleft_cell_b[MAX_CLEFTS];
	int cleft_cell_a_junc[MAX_CLEFTS];
	int cleft_cell_b_junc[MAX_CLEFTS];
	double V_cleft[MAX_CLEFTS];
	double Na_cleft[MAX_CLEFTS];
	double K_cleft[MAX_CLEFTS];
	double d_V_cleft[MAX_CLEFTS];
	double d_Na_cleft[MAX_CLEFTS];
	double d_K_cleft[MAX_CLEFTS];
	double I_Na_cleft[MAX_CLEFTS];
	double I_K_cleft[MAX_CLEFTS];
	double I_tot_cleft[MAX_CLEFTS];
	double I_Na_tot_left_cl[MAX_CLEFTS];
	double I_Na_tot_right_cl[MAX_CLEFTS];
	double I_K_tot_left_cl[MAX_CLEFTS];
	double I_K_tot_right_cl[MAX_CLEFTS];
	double g_cleft[MAX_CLEFTS];
	double width[MAX_CLEFTS];
	double vcl[MAX_CLEFTS];
	double G_gap[MAX_CLEFTS];
	double I_gap[MAX_CLEFTS];
	int size_of_para_matrix;
	int matrix_nnz;
	int csr_row_ptr[MAX_MATRIX_SIZE + 1];
	int csr_col_ind[MAX_MATRIX_NNZ];
	double csr_val[MAX_MATRIX_NNZ];
	double constant[MAX_MATRIX_SIZE];
	double solution[MAX_MATRIX_SIZE];
	double spd_rhs[MAX_MATRIX_SIZE];
	double cg_r[MAX_MATRIX_SIZE];
	double cg_p[MAX_MATRIX_SIZE];
	double cg_Ap[MAX_MATRIX_SIZE];
	double pcg_z[MAX_MATRIX_SIZE];
	double pcg_M_inv[MAX_MATRIX_SIZE];
	double pcg_partial_rz[MAX_PCG_BLOCKS];
	double pcg_partial_aux[MAX_PCG_BLOCKS];
	double cg_dot[2];
	double pcg_rhs_norm2;
	int pcg_converged;
	int pcg_breakdown;
	int pcg_iterations;
	__host__ Cell_Grid();
	__host__ __device__ void compute_I_cleft(int id);
	__host__ __device__ void compute_dion_dt(int idx);
	__host__ __device__ void cleft_update(int id, double dt);
	__host__ __device__ void arrange_constant(int idx);
};
#include "Cell_Grid.cu"
#endif
