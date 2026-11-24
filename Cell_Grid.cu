template <unsigned int NMYOS>
__host__ Cell_Grid<NMYOS>::Cell_Grid()
{
	num_clefts = 0;
	for (int i = 0; i < NMYOS; ++i)
	{
		cell_num_neighbors[i] = 0;
		cell_num_direct_gaps[i] = 0;
		NUMofCells_in_grid[i] = 1.0;
		stim_mask_first[i] = 0;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; ++j)
		{
			Cell_junc_current[j][i] = 0.0;
			Cell_Na_junc_current[j][i] = 0.0;
			Cell_K1_junc_current[j][i] = 0.0;
			NUMofCell_connected[j][i] = 0.0;
			cell_neighbor_cell[j][i] = -1;
			cell_neighbor_cleft[j][i] = -1;
			cell_direct_gap_neighbor[j][i] = -1;
			cell_direct_gap_G[j][i] = 0.0;
		}
	}
	for (int i = 0; i < MAX_CLEFTS; ++i)
	{
		cleft_cell_a[i] = -1;
		cleft_cell_b[i] = -1;
		cleft_cell_a_junc[i] = -1;
		cleft_cell_b_junc[i] = -1;
		I_Na_cleft[i] = 0;
		I_K_cleft[i] = 0;
		I_tot_cleft[i] = 0;
		I_Na_tot_left_cl[i] = 0;
		I_Na_tot_right_cl[i] = 0;
		I_K_tot_left_cl[i] = 0;
		I_K_tot_right_cl[i] = 0;
		V_cleft[i] = 0.0;
		Na_cleft[i] = 140;
		K_cleft[i] = 5.4;
		d_V_cleft[i] = 0.0;
		d_Na_cleft[i] = 0.0;
		d_K_cleft[i] = 0.0;
		g_cleft[i] = 0.0;
		width[i] = 0.0;
		vcl[i] = 0.0;
		G_gap[i] = 0.0;
		I_gap[i] = 0.0;
	}
	size_of_para_matrix = NMYOS;
	matrix_nnz = 0;
	cg_dot[0] = 0.0;
	cg_dot[1] = 0.0;
	for (int i = 0; i < MAX_MATRIX_SIZE; ++i)
	{
		constant[i] = 0.0;
		solution[i] = 0.0;
		spd_rhs[i] = 0.0;
		cg_r[i] = 0.0;
		cg_p[i] = 0.0;
		cg_Ap[i] = 0.0;
		pcg_z[i] = 0.0;
		pcg_M_inv[i] = 1.0;
		csr_row_ptr[i] = 0;
	}
	csr_row_ptr[MAX_MATRIX_SIZE] = 0;
	for (int i = 0; i < MAX_MATRIX_NNZ; ++i)
	{
		csr_col_ind[i] = 0;
		csr_val[i] = 0.0;
	}
	for (int i = 0; i < MAX_PCG_BLOCKS; ++i)
	{
		pcg_partial_rz[i] = 0.0;
		pcg_partial_aux[i] = 0.0;
	}
	pcg_rhs_norm2 = 0.0;
	pcg_converged = 0;
	pcg_breakdown = 0;
	pcg_iterations = 0;
}

template <unsigned int NMYOS>
__host__ __device__ void Cell_Grid<NMYOS>::compute_I_cleft(int id)
{
	double Ko = 5.4;
	double Nao = 140.0;
	if (fabs(V_cleft[id]) < 1e-5)
	{
		I_Na_cleft[id] = 1e-1 * g_cleft[id] / FoRT * (Na_cleft[id] - Nao * exp(-V_cleft[id] * FoRT)) / (Nao + Ko);
		I_K_cleft[id] = 1e-1 * g_cleft[id] / FoRT * (K_cleft[id] - Ko * exp(-V_cleft[id] * FoRT)) / (Nao + Ko);
	}
	else
	{
		I_K_cleft[id] = 1e-1 * g_cleft[id] * V_cleft[id] * (K_cleft[id] - Ko * exp(-V_cleft[id] * FoRT)) / (Nao + Ko) / (1 - exp(-V_cleft[id] * FoRT));
		I_Na_cleft[id] = 1e-1 * g_cleft[id] * V_cleft[id] * (Na_cleft[id] - Nao * exp(-V_cleft[id] * FoRT)) / (Nao + Ko) / (1 - exp(-V_cleft[id] * FoRT));
	}
	I_tot_cleft[id] = I_Na_cleft[id] + I_K_cleft[id];
}

template <unsigned int NMYOS>
__host__ __device__ void Cell_Grid<NMYOS>::compute_dion_dt(int idx)
{
	int id_cleft = idx;
	int cell_a = cleft_cell_a[id_cleft];
	int cell_b = cleft_cell_b[id_cleft];
	int junc_a = cleft_cell_a_junc[id_cleft];
	int junc_b = cleft_cell_b_junc[id_cleft];
	I_Na_tot_left_cl[id_cleft] = Cell_Na_junc_current[junc_a][cell_a];
	I_K_tot_left_cl[id_cleft] = Cell_K1_junc_current[junc_a][cell_a];
	I_Na_tot_right_cl[id_cleft] = Cell_Na_junc_current[junc_b][cell_b];
	I_K_tot_right_cl[id_cleft] = Cell_K1_junc_current[junc_b][cell_b];
	d_Na_cleft[id_cleft] = 1e6 * 1 / (Frdy / 1000 * vcl[id_cleft]) * (I_Na_tot_left_cl[id_cleft] + I_Na_tot_right_cl[id_cleft] - I_Na_cleft[id_cleft]);
	d_K_cleft[id_cleft] = 1e6 * 1 / (Frdy / 1000 * vcl[id_cleft]) * (I_K_tot_left_cl[id_cleft] + I_K_tot_right_cl[id_cleft] - I_K_cleft[id_cleft]);
}

template <unsigned int NMYOS>
__host__ __device__ void Cell_Grid<NMYOS>::cleft_update(int id, double dt)
{
	V_cleft[id] += d_V_cleft[id] * dt;
	Na_cleft[id] += d_Na_cleft[id] * dt;
	K_cleft[id] += d_K_cleft[id] * dt;
	if (Na_cleft[id] < 1e-5)
	{
		Na_cleft[id] = 1e-5;
	}
	if (K_cleft[id] < 1e-5)
	{
		K_cleft[id] = 1e-5;
	}
}

template <unsigned int NMYOS>
__host__ __device__ void Cell_Grid<NMYOS>::arrange_constant(int idx)
{
	if (idx < NMYOS)
	{
		int id_cell = idx;
		double rhs_current = Cell_lat_current[id_cell];
		for (int j = 0; j < cell_num_neighbors[id_cell]; j++)
		{
			int id_cleft = cell_neighbor_cleft[j][id_cell];
			double gap_sign = (cleft_cell_a[id_cleft] == id_cell) ? 1.0 : -1.0;
			rhs_current += gap_sign * I_gap[id_cleft] + Cell_junc_current[j][id_cell];
		}
		for (int j = 0; j < cell_num_direct_gaps[id_cell]; j++)
		{
			int neighbor = cell_direct_gap_neighbor[j][id_cell];
			rhs_current += cell_direct_gap_G[j][id_cell] * (V_Cell[id_cell] - V_Cell[neighbor]);
		}
		constant[idx] = -rhs_current;
		return;
	}
	int id_cleft = idx - NMYOS;
	int cell_a = cleft_cell_a[id_cleft];
	int cell_b = cleft_cell_b[id_cleft];
	int junc_a = cleft_cell_a_junc[id_cleft];
	int junc_b = cleft_cell_b_junc[id_cleft];
	constant[idx] = -(Cell_junc_current[junc_a][cell_a] + Cell_junc_current[junc_b][cell_b] - I_tot_cleft[id_cleft]);
}
