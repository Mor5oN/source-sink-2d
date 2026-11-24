#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template <unsigned int NMYOS>
SpatialTissue<NMYOS>::SpatialTissue(double vertical_junc_radius_input,
									double vertical_Na_share_input,
									double vertical_K_share_input,
									bool vertical_gap_only_mode_input,
									double stagger_lambda_input)
{
	size_of_para_matrix = NMYOS;
	radius = 15.0;
	length = 100.0;
	vertical_junc_radius = vertical_junc_radius_input;
	vertical_Na_share = vertical_Na_share_input;
	vertical_K_share = vertical_K_share_input;
	vertical_gap_only_mode = vertical_gap_only_mode_input;
	stagger_lambda = stagger_lambda_input;
	if (stagger_lambda < 0.0 || stagger_lambda > 0.5)
	{
		fprintf(stderr, "stagger_lambda must satisfy 0 <= value <= 0.5.\n");
		exit(1);
	}
	if (vertical_junc_radius < 0.0)
	{
		fprintf(stderr, "Invalid vertical_junc_radius %f. Expected value >= 0.\n", vertical_junc_radius);
		exit(1);
	}
	if (vertical_Na_share < 0.0 || vertical_Na_share > 1.0 ||
		vertical_K_share < 0.0 || vertical_K_share > 1.0)
	{
		fprintf(stderr, "vertical_Na_share and vertical_K_share must satisfy 0 <= value <= 1.\n");
		exit(1);
	}
	if (vertical_junc_radius == 0.0 && (vertical_Na_share > 0.0 || vertical_K_share > 0.0))
	{
		fprintf(stderr, "vertical_Na_share and vertical_K_share must be 0 when vertical_junc_radius is 0.\n");
		exit(1);
	}
	if (vertical_gap_only_mode && (vertical_Na_share > 0.0 || vertical_K_share > 0.0))
	{
		fprintf(stderr, "vertical_Na_share and vertical_K_share must be 0 in vertical gap-only mode.\n");
		exit(1);
	}
	dimBlock_myo.x = BLOCK_SIZE_X;
	dimBlock_grid.x = BLOCK_SIZE_X;
	dimBlock_cleft.x = BLOCK_SIZE_X;
	dimBlock_matrix.x = BLOCK_SIZE_X;
	dimBlock_myo.y = BLOCK_SIZE_Y;
	dimBlock_grid.y = BLOCK_SIZE_Y;
	dimBlock_cleft.y = BLOCK_SIZE_Y;
	dimBlock_matrix.y = BLOCK_SIZE_Y;
	dimGrid_myo.x = (NMYOS + dimBlock_myo.x - 1) / dimBlock_myo.x;
	dimGrid_grid.x = (NMYOS + dimBlock_grid.x - 1) / dimBlock_grid.x;
	dimGrid_cleft.x = (NMYOS + dimBlock_cleft.x - 1) / dimBlock_cleft.x;
	dimGrid_matrix.x = (size_of_para_matrix + dimBlock_matrix.x - 1) / dimBlock_matrix.x;
	dimGrid_myo.y = 1;
	dimGrid_grid.y = 1;
	dimGrid_cleft.y = 1;
	dimGrid_matrix.y = 1;
	cudaStreamCreate(&MyoStream);
	cudaStreamCreate(&GridStream);
	rabbit_myocyte = new myocyte<NMYOS>();
	cable_grid = new Cell_Grid<NMYOS>();
	build_staggered_topology();
	size_of_para_matrix = cable_grid -> size_of_para_matrix;
	dimGrid_cleft.x = (cable_grid -> num_clefts + dimBlock_cleft.x - 1) / dimBlock_cleft.x;
	dimGrid_matrix.x = (size_of_para_matrix + dimBlock_matrix.x - 1) / dimBlock_matrix.x;
	gpuErrchk(cudaMalloc((void**) &d_rabbit_myocyte, sizeof(myocyte<NMYOS>)));
	gpuErrchk(cudaMalloc((void**) &d_cable_grid, sizeof(Cell_Grid<NMYOS>)));
	gpuErrchk(cudaMalloc((void**) &d_output_voltage, NMYOS * sizeof(double)));
	gpuErrchk(cudaMalloc((void**) &d_pcg_status, 2 * sizeof(int)));

	for (int idx = 0; idx < NMYOS; idx++)
	{
		rabbit_myocyte -> myo_location[idx] = idx;
		V_Cell[idx] = rabbit_myocyte -> y[7][idx];
	}
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::add_cleft_connection(int cell_a, int cell_b)
{
	if (cell_a < 0 || cell_b < 0 || cell_a >= (int)NMYOS || cell_b >= (int)NMYOS || cell_a == cell_b)
	{
		return;
	}
	if (cell_a > cell_b)
	{
		int temp = cell_a;
		cell_a = cell_b;
		cell_b = temp;
	}
	for (int id_cleft = 0; id_cleft < cable_grid -> num_clefts; id_cleft++)
	{
		if (cable_grid -> cleft_cell_a[id_cleft] == cell_a && cable_grid -> cleft_cell_b[id_cleft] == cell_b)
		{
			return;
		}
	}
	int junc_a = cable_grid -> cell_num_neighbors[cell_a];
	int junc_b = cable_grid -> cell_num_neighbors[cell_b];
	if (junc_a >= MAX_CELL_NEIGHBORS || junc_b >= MAX_CELL_NEIGHBORS)
	{
		fprintf(stderr, "Too many neighbors for cell %d or %d. Increase MAX_CELL_NEIGHBORS.\n", cell_a, cell_b);
		exit(1);
	}
	if (cable_grid -> num_clefts >= MAX_CLEFTS)
	{
		fprintf(stderr, "Too many clefts. Increase MAX_CLEFTS.\n");
		exit(1);
	}
	int id_cleft = cable_grid -> num_clefts++;
	cable_grid -> cleft_cell_a[id_cleft] = cell_a;
	cable_grid -> cleft_cell_b[id_cleft] = cell_b;
	cable_grid -> cleft_cell_a_junc[id_cleft] = junc_a;
	cable_grid -> cleft_cell_b_junc[id_cleft] = junc_b;
	cable_grid -> cell_neighbor_cell[junc_a][cell_a] = cell_b;
	cable_grid -> cell_neighbor_cleft[junc_a][cell_a] = id_cleft;
	cable_grid -> cell_neighbor_cell[junc_b][cell_b] = cell_a;
	cable_grid -> cell_neighbor_cleft[junc_b][cell_b] = id_cleft;
	cable_grid -> cell_num_neighbors[cell_a]++;
	cable_grid -> cell_num_neighbors[cell_b]++;
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::add_direct_gap_connection(int cell_a, int cell_b)
{
	if (cell_a < 0 || cell_b < 0 || cell_a >= (int)NMYOS || cell_b >= (int)NMYOS || cell_a == cell_b)
	{
		return;
	}
	for (int j = 0; j < cable_grid -> cell_num_direct_gaps[cell_a]; j++)
	{
		if (cable_grid -> cell_direct_gap_neighbor[j][cell_a] == cell_b)
		{
			return;
		}
	}
	int gap_a = cable_grid -> cell_num_direct_gaps[cell_a];
	int gap_b = cable_grid -> cell_num_direct_gaps[cell_b];
	if (gap_a >= MAX_CELL_NEIGHBORS || gap_b >= MAX_CELL_NEIGHBORS)
	{
		fprintf(stderr, "Too many direct gap neighbors for cell %d or %d. Increase MAX_CELL_NEIGHBORS.\n", cell_a, cell_b);
		exit(1);
	}
	cable_grid -> cell_direct_gap_neighbor[gap_a][cell_a] = cell_b;
	cable_grid -> cell_direct_gap_neighbor[gap_b][cell_b] = cell_a;
	cable_grid -> cell_direct_gap_G[gap_a][cell_a] = 0.0;
	cable_grid -> cell_direct_gap_G[gap_b][cell_b] = 0.0;
	cable_grid -> cell_num_direct_gaps[cell_a]++;
	cable_grid -> cell_num_direct_gaps[cell_b]++;
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::build_staggered_topology()
{
	int rows = MYO_ROWS;
	int cols = MYO_COLS;
	if (rows <= 0 || cols <= 0 || rows * cols != (int)NMYOS)
	{
		fprintf(stderr, "Invalid myocyte grid: MYO_ROWS * MYO_COLS must equal NMYOS (%d * %d != %d).\n", rows, cols, (int)NMYOS);
		exit(1);
	}
	cable_grid -> num_clefts = 0;
	for (int id = 0; id < (int)NMYOS; id++)
	{
		cell_num_neighbors[id] = 0;
		cable_grid -> cell_num_neighbors[id] = 0;
		cable_grid -> cell_num_direct_gaps[id] = 0;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			cable_grid -> cell_neighbor_cell[j][id] = -1;
			cable_grid -> cell_neighbor_cleft[j][id] = -1;
			cable_grid -> cell_direct_gap_neighbor[j][id] = -1;
			cable_grid -> cell_direct_gap_G[j][id] = 0.0;
		}
	}
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			int cell = row * cols + col;
			if (col + 1 < cols)
			{
				int upper_right_row = (col % 2 == 0) ? row - 1 : row;
				int lower_right_row = (col % 2 == 0) ? row : row + 1;
				if (upper_right_row >= 0 && upper_right_row < rows)
				{
					double frac_upper = (upper_right_row == row) ? (1.0 - stagger_lambda) : stagger_lambda;
					if (frac_upper > 1e-12)
					{
						add_cleft_connection(cell, upper_right_row * cols + col + 1);
					}
				}
				if (lower_right_row >= 0 && lower_right_row < rows && lower_right_row != upper_right_row)
				{
					double frac_lower = (lower_right_row == row) ? (1.0 - stagger_lambda) : stagger_lambda;
					if (frac_lower > 1e-12)
					{
						add_cleft_connection(cell, lower_right_row * cols + col + 1);
					}
				}
			}
		}
	}
	if (vertical_gap_only_mode || vertical_junc_radius > 0.0)
	{
		if (rows < 2)
		{
			fprintf(stderr, "Vertical connections require at least 2 MYO_ROWS.\n");
			exit(1);
		}
		for (int row = 0; row + 1 < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				if (vertical_gap_only_mode)
				{
					add_direct_gap_connection(row * cols + col, (row + 1) * cols + col);
				}
				else
				{
					add_cleft_connection(row * cols + col, (row + 1) * cols + col);
				}
			}
		}
	}
	for (int id = 0; id < (int)NMYOS; id++)
	{
		cell_num_neighbors[id] = cable_grid -> cell_num_neighbors[id];
		if (cell_num_neighbors[id] > MAX_CELL_NEIGHBORS)
		{
			fprintf(stderr, "Unexpected neighbor count for staggered topology: cell %d has %d neighbors.\n", id, cell_num_neighbors[id]);
			exit(1);
		}
	}
	cable_grid -> size_of_para_matrix = NMYOS + cable_grid -> num_clefts;
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::arrange_center_stimulus_region(int num_stimulated_cells, int shape_type)
{
	int rows = MYO_ROWS;
	int cols = MYO_COLS;
	if (rows <= 0 || cols <= 0 || rows * cols != (int)NMYOS)
	{
		fprintf(stderr, "Invalid myocyte grid: MYO_ROWS * MYO_COLS must equal NMYOS (%d * %d != %d).\n", rows, cols, (int)NMYOS);
		exit(1);
	}
	for (int id = 0; id < (int)NMYOS; id++)
	{
		cable_grid -> stim_mask_first[id] = 0;
	}
	if (num_stimulated_cells <= 0)
	{
		return;
	}
	if (num_stimulated_cells > (int)NMYOS)
	{
		num_stimulated_cells = NMYOS;
	}

	int col_mid = cols / 2;
	int row_mid = rows / 2;
	double x_mid = col_mid * (sqrt(3.0) / 2.0);
	double y_mid = row_mid + 0.5 * (col_mid % 2);

	struct CellDist {
		int id;
		double dist;
	};

	std::vector<CellDist> cells(NMYOS);
	for (int id = 0; id < (int)NMYOS; id++)
	{
		int r = id / cols;
		int c = id % cols;
		double x = c * (sqrt(3.0) / 2.0);
		double y = r + 0.5 * (c % 2);
		double dx = x - x_mid;
		double dy = y - y_mid;

		double dist = 0.0;
		if (shape_type == 0)
		{
			dist = sqrt(dx * dx + dy * dy);
		}
		else if (shape_type == 1)
		{
			dist = fmax(fabs(dx), fabs(dy));
		}
		else if (shape_type == 2)
		{
			dist = fabs(dx) + fabs(dy);
		}
		else
		{
			dist = sqrt(dx * dx + dy * dy);
		}
		cells[id].id = id;
		cells[id].dist = dist;
	}

	std::sort(cells.begin(), cells.end(), [](const CellDist& a, const CellDist& b) {
		if (fabs(a.dist - b.dist) < 1e-9) {
			return a.id < b.id;
		}
		return a.dist < b.dist;
	});

	for (int i = 0; i < num_stimulated_cells; i++)
	{
		cable_grid -> stim_mask_first[cells[i].id] = 1;
	}

	printf("Center stimulation configured: shape=%d, target_cells=%d, actual_stimulated=%d, max_dist=%.4f\n",
	       shape_type, num_stimulated_cells, num_stimulated_cells, cells[num_stimulated_cells - 1].dist);
}

template <unsigned int NMYOS>
SpatialTissue<NMYOS>::~SpatialTissue()
{
	delete rabbit_myocyte;
	delete cable_grid;
	cudaStreamDestroy(MyoStream);
	cudaStreamDestroy(GridStream);
	cudaFree(d_rabbit_myocyte);
	cudaFree(d_cable_grid);
	cudaFree(d_output_voltage);
	cudaFree(d_pcg_status);
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::step(double dt, double &Istim, double in_time, int stimulus_id)
{
	Compute_Myocyte_ODE<<<dimGrid_myo, dimBlock_myo, 0, MyoStream>>>(dt, d_rabbit_myocyte, d_cable_grid, Istim, in_time, stimulus_id);
	gpuErrchk(cudaDeviceSynchronize());
	Compute_Cleft_ODE<<<dimGrid_cleft, dimBlock_cleft, 0, GridStream>>>(d_cable_grid);
	Arrange_Linear_Constant<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
	Prepare_PCG<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
	Finalize_Initial_PCG<<<1, PCG_THREADS_PER_BLOCK, 0, GridStream>>>(d_cable_grid, dimGrid_matrix.x);
	int pcg_status[2] = {0, 0};
	for (int iter = 0; iter < PCG_MAX_ITER; iter++)
	{
		PCG_MatVec_Direction_And_Dot<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
		PCG_Finalize_Direction_Dot<<<1, PCG_THREADS_PER_BLOCK, 0, GridStream>>>(d_cable_grid, dimGrid_matrix.x);
		PCG_Update_X_R_Z_And_Dot<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
		PCG_Finalize_RZ_And_Check<<<1, PCG_THREADS_PER_BLOCK, 0, GridStream>>>(d_cable_grid, dimGrid_matrix.x);
		PCG_Update_Direction<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
#if PCG_CHECK_INTERVAL > 0
		if (((iter + 1) % PCG_CHECK_INTERVAL == 0) || (iter + 1 == PCG_MAX_ITER))
		{
			Copy_PCG_Status<<<1, 1, 0, GridStream>>>(d_cable_grid, d_pcg_status);
			gpuErrchk(cudaMemcpyAsync(pcg_status, d_pcg_status, 2 * sizeof(int), cudaMemcpyDeviceToHost, GridStream));
			gpuErrchk(cudaStreamSynchronize(GridStream));
			if (pcg_status[0] || pcg_status[1])
			{
				break;
			}
		}
#endif
	}
	gpuErrchk(cudaDeviceSynchronize());
	Assignment_myocyte_dvdt<<<dimGrid_matrix, dimBlock_matrix, 0, MyoStream>>>(d_rabbit_myocyte, d_cable_grid);
	Assignment_Cleft_dvdt<<<dimGrid_matrix, dimBlock_matrix, 0, GridStream>>>(d_cable_grid);
	Myocyte_Update<<<dimGrid_myo, dimBlock_myo, 0, MyoStream>>>(dt, d_rabbit_myocyte);
	Cleft_update<<<dimGrid_cleft, dimBlock_cleft, 0, GridStream>>>(dt, d_cable_grid);
	gpuErrchk(cudaDeviceSynchronize());
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::copy_voltage_to_host()
{
	Copy_Voltage_For_Output<<<dimGrid_myo, dimBlock_myo>>>(d_rabbit_myocyte, d_output_voltage);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(cable_grid -> V_Cell, d_output_voltage, NMYOS * sizeof(double), cudaMemcpyDeviceToHost));
}

template <unsigned int NMYOS>
__global__ void Compute_Myocyte_ODE(double dt, myocyte<NMYOS>* rabbit_myocyte,
									Cell_Grid<NMYOS>* cable_grid,
									double Istim,
									double in_time,
									int stimulus_id)
{
	double Istim0;
	int location;
	int stimulated;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= NMYOS)
		return;
	rabbit_myocyte -> eccODEfile(id, dt, in_time);
	for (int j = 0; j < rabbit_myocyte -> cell_num_neighbors[id]; j++)
	{
		int id_cleft = cable_grid -> cell_neighbor_cleft[j][id];
		rabbit_myocyte -> V_myo_junc[j][id] = rabbit_myocyte -> y[7][id] - cable_grid -> V_cleft[id_cleft];
		rabbit_myocyte -> Nao[j][id] = cable_grid -> Na_cleft[id_cleft];
		rabbit_myocyte -> Ko[j][id] = cable_grid -> K_cleft[id_cleft];
		rabbit_myocyte -> rabbit_juncODEfile(id, j, dt);
	}
	stimulated = (stimulus_id == 1) ? cable_grid -> stim_mask_first[id] : 0;
	Istim0 = stimulated ? Istim : 0.0;
	rabbit_myocyte -> rabbit_covert_to_Current(id, Istim0);
	location = rabbit_myocyte -> myo_location[id];
	cable_grid -> V_Cell[location] = rabbit_myocyte -> y[7][id];
	cable_grid -> Cell_lat_current[location] = rabbit_myocyte -> I_tot_lateral[id];
	for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
	{
		cable_grid -> Cell_junc_current[j][location] = rabbit_myocyte -> I_tot_myo_cl[j][id];
		cable_grid -> Cell_Na_junc_current[j][location] = rabbit_myocyte -> I_Na_tot_myo_cl[j][id];
		cable_grid -> Cell_K1_junc_current[j][location] = rabbit_myocyte -> I_K1_tot_myo_cl[j][id];
	}
}

template <unsigned int NMYOS>
__global__ void Compute_Cleft_ODE(Cell_Grid<NMYOS>* cable_grid)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= cable_grid -> num_clefts)
		return;
	cable_grid -> compute_I_cleft(id);
	int cell_a = cable_grid -> cleft_cell_a[id];
	int cell_b = cable_grid -> cleft_cell_b[id];
	cable_grid -> I_gap[id] = cable_grid -> G_gap[id] * (cable_grid -> V_Cell[cell_a] - cable_grid -> V_Cell[cell_b]);
}

template <unsigned int NMYOS>
__global__ void Arrange_Linear_Constant(Cell_Grid<NMYOS>* cable_grid)
{
	int size_of_para_matrix = cable_grid -> size_of_para_matrix;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size_of_para_matrix)
		return;
	cable_grid -> arrange_constant(idx);
}

template <unsigned int NMYOS>
__global__ void Prepare_PCG(Cell_Grid<NMYOS>* cable_grid)
{
	__shared__ double partial_rz[PCG_THREADS_PER_BLOCK];
	__shared__ double partial_rhs[PCG_THREADS_PER_BLOCK];
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double rz = 0.0;
	double rhs_norm2 = 0.0;
	if (blockIdx.x == 0 && tid == 0)
	{
		cable_grid -> pcg_converged = 0;
		cable_grid -> pcg_breakdown = 0;
		cable_grid -> pcg_iterations = 0;
	}
	if (idx < cable_grid -> size_of_para_matrix)
	{
		double rhs = (idx < NMYOS) ? cable_grid -> constant[idx] : -cable_grid -> constant[idx];
		double value = 0.0;
		for (int nz = cable_grid -> csr_row_ptr[idx]; nz < cable_grid -> csr_row_ptr[idx + 1]; nz++)
		{
			value += cable_grid -> csr_val[nz] * cable_grid -> solution[cable_grid -> csr_col_ind[nz]];
		}
		double r = rhs - value;
		double z = cable_grid -> pcg_M_inv[idx] * r;
		cable_grid -> spd_rhs[idx] = rhs;
		cable_grid -> cg_Ap[idx] = value;
		cable_grid -> cg_r[idx] = r;
		cable_grid -> pcg_z[idx] = z;
		cable_grid -> cg_p[idx] = z;
		rz = r * z;
		rhs_norm2 = rhs * rhs;
	}
	partial_rz[tid] = rz;
	partial_rhs[tid] = rhs_norm2;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial_rz[tid] += partial_rz[tid + stride];
			partial_rhs[tid] += partial_rhs[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		cable_grid -> pcg_partial_rz[blockIdx.x] = partial_rz[0];
		cable_grid -> pcg_partial_aux[blockIdx.x] = partial_rhs[0];
	}
}

template <unsigned int NMYOS>
__global__ void Finalize_Initial_PCG(Cell_Grid<NMYOS>* cable_grid, int num_blocks)
{
	__shared__ double partial_rz[PCG_THREADS_PER_BLOCK];
	__shared__ double partial_rhs[PCG_THREADS_PER_BLOCK];
	int tid = threadIdx.x;
	double rz = 0.0;
	double rhs_norm2 = 0.0;
	for (int i = tid; i < num_blocks; i += blockDim.x)
	{
		rz += cable_grid -> pcg_partial_rz[i];
		rhs_norm2 += cable_grid -> pcg_partial_aux[i];
	}
	partial_rz[tid] = rz;
	partial_rhs[tid] = rhs_norm2;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial_rz[tid] += partial_rz[tid + stride];
			partial_rhs[tid] += partial_rhs[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		double threshold = PCG_TOL * PCG_TOL * fmax(partial_rhs[0], 1.0);
		cable_grid -> cg_dot[0] = partial_rz[0];
		cable_grid -> cg_dot[1] = 0.0;
		cable_grid -> pcg_rhs_norm2 = partial_rhs[0];
		if (!finite_value(partial_rz[0]) || partial_rz[0] < 0.0 || !finite_value(partial_rhs[0]))
		{
			cable_grid -> pcg_breakdown = 1;
		}
		else if (partial_rz[0] <= threshold)
		{
			cable_grid -> pcg_converged = 1;
		}
	}
}

template <unsigned int NMYOS>
__global__ void PCG_MatVec_Direction_And_Dot(Cell_Grid<NMYOS>* cable_grid)
{
	__shared__ double partial[PCG_THREADS_PER_BLOCK];
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double pAp = 0.0;
	if (cable_grid -> pcg_converged || cable_grid -> pcg_breakdown)
	{
		return;
	}
	if (idx < cable_grid -> size_of_para_matrix)
	{
		double value = 0.0;
		for (int nz = cable_grid -> csr_row_ptr[idx]; nz < cable_grid -> csr_row_ptr[idx + 1]; nz++)
		{
			value += cable_grid -> csr_val[nz] * cable_grid -> cg_p[cable_grid -> csr_col_ind[nz]];
		}
		cable_grid -> cg_Ap[idx] = value;
		pAp = cable_grid -> cg_p[idx] * value;
	}
	partial[tid] = pAp;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial[tid] += partial[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		cable_grid -> pcg_partial_aux[blockIdx.x] = partial[0];
	}
}

template <unsigned int NMYOS>
__global__ void PCG_Finalize_Direction_Dot(Cell_Grid<NMYOS>* cable_grid, int num_blocks)
{
	__shared__ double partial[PCG_THREADS_PER_BLOCK];
	int tid = threadIdx.x;
	if (cable_grid -> pcg_converged || cable_grid -> pcg_breakdown)
		return;
	double pAp = 0.0;
	for (int i = tid; i < num_blocks; i += blockDim.x)
	{
		pAp += cable_grid -> pcg_partial_aux[i];
	}
	partial[tid] = pAp;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial[tid] += partial[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		if (!finite_value(partial[0]) || partial[0] <= 1e-300)
		{
			cable_grid -> pcg_breakdown = 1;
		}
		else
		{
			cable_grid -> cg_dot[1] = partial[0];
		}
	}
}

template <unsigned int NMYOS>
__global__ void PCG_Update_X_R_Z_And_Dot(Cell_Grid<NMYOS>* cable_grid)
{
	__shared__ double partial[PCG_THREADS_PER_BLOCK];
	const unsigned int tid = threadIdx.x;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double rz = 0.0;
	if (cable_grid -> pcg_converged || cable_grid -> pcg_breakdown)
		return;
	double alpha = cable_grid -> cg_dot[0] / cable_grid -> cg_dot[1];
	if (!finite_value(alpha))
	{
		if (idx == 0)
		{
			cable_grid -> pcg_breakdown = 1;
		}
		return;
	}
	if (idx < cable_grid -> size_of_para_matrix)
	{
		cable_grid -> solution[idx] += alpha * cable_grid -> cg_p[idx];
		double r = cable_grid -> cg_r[idx] - alpha * cable_grid -> cg_Ap[idx];
		double z = cable_grid -> pcg_M_inv[idx] * r;
		cable_grid -> cg_r[idx] = r;
		cable_grid -> pcg_z[idx] = z;
		rz = r * z;
	}
	partial[tid] = rz;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial[tid] += partial[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		cable_grid -> pcg_partial_rz[blockIdx.x] = partial[0];
	}
}

template <unsigned int NMYOS>
__global__ void PCG_Finalize_RZ_And_Check(Cell_Grid<NMYOS>* cable_grid, int num_blocks)
{
	__shared__ double partial[PCG_THREADS_PER_BLOCK];
	int tid = threadIdx.x;
	if (cable_grid -> pcg_converged || cable_grid -> pcg_breakdown)
		return;
	double rz = 0.0;
	for (int i = tid; i < num_blocks; i += blockDim.x)
	{
		rz += cable_grid -> pcg_partial_rz[i];
	}
	partial[tid] = rz;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partial[tid] += partial[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		double old_rz = cable_grid -> cg_dot[0];
		double threshold = PCG_TOL * PCG_TOL * fmax(cable_grid -> pcg_rhs_norm2, 1.0);
		cable_grid -> pcg_iterations += 1;
		if (!finite_value(partial[0]) || partial[0] < 0.0 || fabs(old_rz) <= 1e-300)
		{
			cable_grid -> pcg_breakdown = 1;
			return;
		}
		cable_grid -> cg_dot[0] = partial[0];
		if (partial[0] <= threshold)
		{
			cable_grid -> pcg_converged = 1;
			return;
		}
		double beta = partial[0] / old_rz;
		if (!finite_value(beta))
		{
			cable_grid -> pcg_breakdown = 1;
			return;
		}
		cable_grid -> cg_dot[1] = beta;
	}
}

template <unsigned int NMYOS>
__global__ void PCG_Update_Direction(Cell_Grid<NMYOS>* cable_grid)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (cable_grid -> pcg_converged || cable_grid -> pcg_breakdown || idx >= cable_grid -> size_of_para_matrix)
		return;
	double beta = cable_grid -> cg_dot[1];
	cable_grid -> cg_p[idx] = cable_grid -> pcg_z[idx] + beta * cable_grid -> cg_p[idx];
}

template <unsigned int NMYOS>
__global__ void Assignment_myocyte_dvdt(myocyte<NMYOS>* rabbit_myocyte,
										Cell_Grid<NMYOS>* cable_grid)
{
	int size_of_para_matrix = cable_grid -> size_of_para_matrix;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= size_of_para_matrix)
		return;
	if (idx < NMYOS)
	{
		rabbit_myocyte -> ydot[7][idx] = cable_grid -> solution[id];
	}
}

template <unsigned int NMYOS>
__global__ void Assignment_Cleft_dvdt(Cell_Grid<NMYOS>* cable_grid)
{
	int size_of_para_matrix = cable_grid -> size_of_para_matrix;
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int id  = idx;
	if (idx >= size_of_para_matrix)
		return;
	if (idx >= NMYOS)
	{
		int id_cleft = idx - NMYOS;
		cable_grid -> d_V_cleft[id_cleft] = cable_grid -> solution[id];
	}
}

template <unsigned int NMYOS>
__global__ void Copy_Voltage_For_Output(myocyte<NMYOS>* rabbit_myocyte, double* output_voltage)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= NMYOS)
		return;
	output_voltage[idx] = rabbit_myocyte -> y[7][idx];
}

template <unsigned int NMYOS>
__global__ void Copy_PCG_Status(Cell_Grid<NMYOS>* cable_grid, int* pcg_status)
{
	pcg_status[0] = cable_grid -> pcg_converged;
	pcg_status[1] = cable_grid -> pcg_breakdown;
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::build_sparse_matrix()
{
	int nnz = 0;
	int size = cable_grid -> size_of_para_matrix;
	cable_grid -> csr_row_ptr[0] = 0;
	for (int idx = 0; idx < size; idx++)
	{
		if (idx < NMYOS)
		{
			int id_cell = idx;
			if (nnz + 1 + cell_num_neighbors[id_cell] > Cell_Grid<NMYOS>::MAX_MATRIX_NNZ)
			{
				fprintf(stderr, "CSR matrix exceeds MAX_MATRIX_NNZ.\n");
				exit(1);
			}
			double diagonal = C_cell_lat[id_cell];
			cable_grid -> csr_col_ind[nnz] = idx;
			int diagonal_nz = nnz;
			cable_grid -> csr_val[nnz] = diagonal;
			nnz++;
			for (int j = 0; j < cell_num_neighbors[id_cell]; j++)
			{
				int id_cleft = cable_grid -> cell_neighbor_cleft[j][id_cell];
				double c_junc = NUMofCell_connected[j][id_cell] * C_cell_junc[j][id_cell];
				diagonal += c_junc;
				cable_grid -> csr_col_ind[nnz] = NMYOS + id_cleft;
				cable_grid -> csr_val[nnz] = -c_junc;
				nnz++;
			}
			cable_grid -> csr_val[diagonal_nz] = diagonal;
			cable_grid -> pcg_M_inv[idx] = (fabs(diagonal) > 1e-300) ? 1.0 / diagonal : 1.0;
		}
		else
		{
			if (nnz + 3 > Cell_Grid<NMYOS>::MAX_MATRIX_NNZ)
			{
				fprintf(stderr, "CSR matrix exceeds MAX_MATRIX_NNZ.\n");
				exit(1);
			}
			int id_cleft = idx - NMYOS;
			int cell_a = cable_grid -> cleft_cell_a[id_cleft];
			int cell_b = cable_grid -> cleft_cell_b[id_cleft];
			int junc_a = cable_grid -> cleft_cell_a_junc[id_cleft];
			int junc_b = cable_grid -> cleft_cell_b_junc[id_cleft];
			double c_a = C_cell_junc[junc_a][cell_a];
			double c_b = C_cell_junc[junc_b][cell_b];
			cable_grid -> csr_col_ind[nnz] = cell_a;
			cable_grid -> csr_val[nnz] = -c_a;
			nnz++;
			cable_grid -> csr_col_ind[nnz] = cell_b;
			cable_grid -> csr_val[nnz] = -c_b;
			nnz++;
			cable_grid -> csr_col_ind[nnz] = idx;
			double diagonal = c_a + c_b;
			cable_grid -> csr_val[nnz] = diagonal;
			nnz++;
			cable_grid -> pcg_M_inv[idx] = (fabs(diagonal) > 1e-300) ? 1.0 / diagonal : 1.0;
		}
		cable_grid -> csr_row_ptr[idx + 1] = nnz;
	}
	cable_grid -> matrix_nnz = nnz;
	for (int i = 0; i < size; i++)
	{
		cable_grid -> solution[i] = 0.0;
	}
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::arrange_cells_surface_and_Capacitance(double f_Ina_cleft_input, double f_Ina_lateral_input)
{
	f_Ina_cleft = f_Ina_cleft_input;
	f_Ina_lateral = f_Ina_lateral_input;
	A_m = 2 * pi * radius * length +
		  2 * pi * radius * radius;
	A_myo_j = pi * radius * radius;
	A_myo_j_vertical = pi * vertical_junc_radius * vertical_junc_radius;
	const int rows = MYO_ROWS;
	const int cols = MYO_COLS;
	int id;
	for (int idx = 0; idx < NMYOS; idx++)
	{
		id = idx;
		int row = id / cols;
		int col = id % cols;
		bool is_half_cell = rows > 1 &&
			((row == 0 && col % 2 == 0) ||
			 (row == rows - 1 && col % 2 == 1));
		double cell_area_scale = is_half_cell ? (1.0 - stagger_lambda) : 1.0;
		cell_num_neighbors[id] = cable_grid -> cell_num_neighbors[id];
		NUMofCells_in_grid[id] = cell_area_scale;
		int horizontal_count = 0;
		int vertical_count = 0;
		for (int j = 0; j < cell_num_neighbors[id]; j++)
		{
			int neighbor = cable_grid -> cell_neighbor_cell[j][id];
			if (neighbor >= 0 && neighbor % cols == col)
			{
				vertical_count++;
			}
			else
			{
				horizontal_count++;
			}
		}
		double vertical_area = A_myo_j_vertical;
		double total_junction_area = 0.0;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			if (j < cell_num_neighbors[id])
			{
				int neighbor = cable_grid -> cell_neighbor_cell[j][id];
				bool is_vertical_connection = neighbor >= 0 && neighbor % cols == col;
				NUMofCell_connected[j][id] = 1.0;
				if (is_vertical_connection)
				{
					A_cell_junc[j][id] = vertical_area;
				}
				else
				{
					double junc_fraction = get_horizontal_junction_fraction(id, neighbor);
					A_cell_junc[j][id] = junc_fraction * A_myo_j;
				}
				total_junction_area += A_cell_junc[j][id];
			}
			else
			{
				NUMofCell_connected[j][id] = 0.0;
				A_cell_junc[j][id] = 0.0;
			}
		}
		A_cell_lat[id] = cell_area_scale * A_m - total_junction_area;
		if (A_cell_lat[id] <= 0.0)
		{
			fprintf(stderr, "Invalid lateral membrane area for cell %d: scale=%f horizontal=%d vertical=%d junction_area=%f.\n",
					id, cell_area_scale, horizontal_count, vertical_count, total_junction_area);
			exit(1);
		}
	}

	C_m = 1e-8;
	for (int idx = 0; idx < NMYOS; idx++)
	{
		id = idx;
		C_cell_lat[id] = A_cell_lat[id] * C_m;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			C_cell_junc[j][id] = A_cell_junc[j][id] * C_m;
		}
	}
	for (int idx = 0; idx < NMYOS; idx++)
	{
		id = idx;
		int col = id % cols;
		double horizontal_junction_area = 0.0;
		double vertical_junction_area = 0.0;
		for (int j = 0; j < cell_num_neighbors[id]; j++)
		{
			int neighbor = cable_grid -> cell_neighbor_cell[j][id];
			bool is_vertical_connection = neighbor >= 0 && neighbor % cols == col;
			if (is_vertical_connection)
			{
				vertical_junction_area += A_cell_junc[j][id] * NUMofCell_connected[j][id];
			}
			else
			{
				horizontal_junction_area += A_cell_junc[j][id] * NUMofCell_connected[j][id];
			}
		}
		double junction_area = horizontal_junction_area + vertical_junction_area;
		double total_area = A_cell_lat[id] + junction_area;
		double lateral_Ina_share = f_Ina_lateral * (1.0 - vertical_Na_share);
		double lateral_K1_share = (1.0 - f_k1) * (1.0 - vertical_K_share);
		double vertical_Ina_share = f_Ina_lateral * vertical_Na_share;
		double vertical_K1_share = (1.0 - f_k1) * vertical_K_share;
		ryo_f_Ina_lat[id] = lateral_Ina_share * total_area / A_cell_lat[id];
		ryo_f_k1_lat[id] = lateral_K1_share * total_area / A_cell_lat[id];
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			if (j < cell_num_neighbors[id])
			{
				int neighbor = cable_grid -> cell_neighbor_cell[j][id];
				bool is_vertical_connection = neighbor >= 0 && neighbor % cols == col;
				if (is_vertical_connection)
				{
					ryo_f_Ina_junc[j][id] = (vertical_junction_area > 0.0) ? vertical_Ina_share * total_area / vertical_junction_area : 0.0;
					ryo_f_k1_junc[j][id] = (vertical_junction_area > 0.0) ? vertical_K1_share * total_area / vertical_junction_area : 0.0;
				}
				else
				{
					ryo_f_Ina_junc[j][id] = (horizontal_junction_area > 0.0) ? f_Ina_cleft * total_area / horizontal_junction_area : 0.0;
					ryo_f_k1_junc[j][id] = (horizontal_junction_area > 0.0) ? f_k1 * total_area / horizontal_junction_area : 0.0;
				}
			}
			else
			{
				ryo_f_Ina_junc[j][id] = 0.0;
				ryo_f_k1_junc[j][id] = 0.0;
			}
		}
	}
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::arrange_cleft_parameter(double *g_cleft_control,
												   double *horizontal_width_of_cleft_myo,
												   double *horizontal_G_gap_myo,
												   double *vertical_width_of_cleft_myo,
												   double *vertical_G_gap_myo)
{
	int id_cleft;
	const int cols = MYO_COLS;
	for (int idx = 0; idx < cable_grid -> num_clefts; idx++)
	{
		id_cleft = idx;
		int cell_a = cable_grid -> cleft_cell_a[id_cleft];
		int cell_b = cable_grid -> cleft_cell_b[id_cleft];
		int junc_a = cable_grid -> cleft_cell_a_junc[id_cleft];
		int junc_b = cable_grid -> cleft_cell_b_junc[id_cleft];
		bool is_vertical_connection = cell_a % cols == cell_b % cols;
		double effective_junction_area = fmin(A_cell_junc[junc_a][cell_a], A_cell_junc[junc_b][cell_b]);
		if (effective_junction_area <= 0.0)
		{
			fprintf(stderr, "Invalid junction area for cleft %d between cell %d and cell %d.\n",
					id_cleft, cell_a, cell_b);
			exit(1);
		}
		if (is_vertical_connection)
		{
			cable_grid -> G_gap[id_cleft] = vertical_G_gap_myo[0];
			cable_grid -> width[id_cleft] = 0.001 * vertical_width_of_cleft_myo[0];
			cable_grid -> vcl[id_cleft] = cable_grid -> width[id_cleft] * effective_junction_area;
			cable_grid -> g_cleft[id_cleft] = 8 * pi * cable_grid -> vcl[id_cleft] / (effective_junction_area * 150) / g_cleft_control[0];
		}
		else
		{
			double junc_fraction = get_horizontal_junction_fraction(cell_a, cell_b);
			cable_grid -> G_gap[id_cleft] = junc_fraction * horizontal_G_gap_myo[0];
			cable_grid -> width[id_cleft] = 0.001 * horizontal_width_of_cleft_myo[0];
			cable_grid -> vcl[id_cleft] = cable_grid -> width[id_cleft] * effective_junction_area;
			double base_half_disc_area = JUNCTION_AREA_FRACTION * pi * radius * radius;
			cable_grid -> g_cleft[id_cleft] = 8 * pi * cable_grid -> vcl[id_cleft] / (base_half_disc_area * 150) / g_cleft_control[0];
		}
	}
	for (int id_cell = 0; id_cell < (int)NMYOS; id_cell++)
	{
		for (int j = 0; j < cable_grid -> cell_num_direct_gaps[id_cell]; j++)
		{
			cable_grid -> cell_direct_gap_G[j][id_cell] = vertical_G_gap_myo[0];
		}
	}
	deliver_parameter_to_class();
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::deliver_parameter_to_class()
{
	int id, location;
	for (int idx = 0; idx < NMYOS; idx++)
	{
		id = idx;
		location = rabbit_myocyte -> myo_location[id];
		V_Cell[location] = rabbit_myocyte -> y[7][id];
		rabbit_myocyte -> A_cell_lat[id] = A_cell_lat[location];
		rabbit_myocyte -> C_cell_lat[id] = C_cell_lat[location];
		rabbit_myocyte -> ryo_f_Ina_lat[id] = ryo_f_Ina_lat[location];
		rabbit_myocyte -> ryo_f_k1_lat[id] = ryo_f_k1_lat[location];
		rabbit_myocyte -> cell_num_neighbors[id] = cell_num_neighbors[location];
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			rabbit_myocyte -> A_cell_junc[j][id] = A_cell_junc[j][location];
			rabbit_myocyte -> C_cell_junc[j][id] = C_cell_junc[j][location];
			rabbit_myocyte -> NUMofCell_connected[j][id] = NUMofCell_connected[j][location];
			rabbit_myocyte -> ryo_f_Ina_junc[j][id] = ryo_f_Ina_junc[j][location];
			rabbit_myocyte -> ryo_f_k1_junc[j][id] = ryo_f_k1_junc[j][location];
		}
	}
	for (int idx = 0; idx < NMYOS; idx++)
	{
		id = idx;
		cable_grid -> V_Cell[id] = V_Cell[id];
		cable_grid -> NUMofCells_in_grid[id] = NUMofCells_in_grid[id];
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			cable_grid -> NUMofCell_connected[j][id] = NUMofCell_connected[j][id];
		}
	}
}

template <unsigned int NMYOS>
void SpatialTissue<NMYOS>::cpu_copy_gpu()
{
	gpuErrchk(cudaMemcpy(d_rabbit_myocyte, rabbit_myocyte, sizeof(myocyte<NMYOS>), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_cable_grid, cable_grid, sizeof(Cell_Grid<NMYOS>), cudaMemcpyHostToDevice));
}
