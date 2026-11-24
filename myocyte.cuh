#ifndef MYOCYTE_H
#define MYOCYTE_H

#ifndef FoRT
#define FoRT (96485.0 / 8314.0 / 310.0)
#endif

#ifndef MAX_CELL_NEIGHBORS
#define MAX_CELL_NEIGHBORS 6
#endif

template <unsigned int NMYOS>
class myocyte
{
public:
	double y[8][NMYOS];
	int myo_location[NMYOS];
	double A_cell_lat[NMYOS];
	double A_cell_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double C_cell_lat[NMYOS];
	double C_cell_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double NUMofCell_connected[MAX_CELL_NEIGHBORS][NMYOS];
	int cell_num_neighbors[NMYOS];
	double ryo_f_Ina_lat[NMYOS];
	double ryo_f_k1_lat[NMYOS];
	double ryo_f_Ina_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double ryo_f_k1_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double ydot[8][NMYOS];
	double i_Na_lateral[NMYOS];
	double i_K1_lateral[NMYOS];
	double i_comp_lateral[NMYOS];
	double I_tot_lateral[NMYOS];

	double m_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double h_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double j_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double V_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double Nao[MAX_CELL_NEIGHBORS][NMYOS];
	double Ko[MAX_CELL_NEIGHBORS][NMYOS];
	double d_m_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double d_h_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];
	double d_j_myo_junc[MAX_CELL_NEIGHBORS][NMYOS];

	double i_Na_myo_cl[MAX_CELL_NEIGHBORS][NMYOS];
	double i_K1_myo_cl[MAX_CELL_NEIGHBORS][NMYOS];
	double I_tot_myo_cl[MAX_CELL_NEIGHBORS][NMYOS];
	double I_Na_tot_myo_cl[MAX_CELL_NEIGHBORS][NMYOS];
	double I_K1_tot_myo_cl[MAX_CELL_NEIGHBORS][NMYOS];

	__host__ myocyte();
	__host__ __device__ void eccODEfile(int id, double dt, double in_time);
	__host__ __device__ void rabbit_juncODEfile(int id, int j, double dt);
	__host__ __device__ void rabbit_covert_to_Current(int id, double Istim);
	__host__ __device__ void rabbit_update(int id, double dt);
};

#include "myocyte.cu"
#endif
