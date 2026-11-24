template <unsigned int NMYOS>
__host__ myocyte<NMYOS>::myocyte()
{
	for (int i = 0; i < NMYOS; i++)
	{
		for (int k = 0; k < 8; k++)
		{
			ydot[k][i] = 0.0;
		}
		y[0][i] = 0.0017;
		y[1][i] = 0.9832;
		y[2][i] = 0.995484;
		y[3][i] = 0.000003;
		y[4][i] = 1.0000;
		y[5][i] = 0.0057;
		y[6][i] = 0.0002;
		y[7][i] = -83.644872;

		cell_num_neighbors[i] = 0;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			m_myo_junc[j][i] = 0.0017;
			h_myo_junc[j][i] = 0.9832;
			j_myo_junc[j][i] = 0.995484;
			A_cell_junc[j][i] = 0.0;
			C_cell_junc[j][i] = 0.0;
			NUMofCell_connected[j][i] = 0.0;
			ryo_f_Ina_junc[j][i] = 0.0;
			ryo_f_k1_junc[j][i] = 0.0;
		}
		I_tot_lateral[i] = 0.0;
		for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
		{
			i_Na_myo_cl[j][i] = 0.0;
			i_K1_myo_cl[j][i] = 0.0;
			I_tot_myo_cl[j][i] = 0.0;
			I_Na_tot_myo_cl[j][i] = 0.0;
			I_K1_tot_myo_cl[j][i] = 0.0;
		}
	}
}

template <unsigned int NMYOS>
__host__ __device__ void myocyte<NMYOS>::eccODEfile(int id, double dt, double in_time)
{
	double RTF_mV = (1.0 / FoRT);
	double V = y[7][id];
	double Cai = y[6][id];
	double m = y[0][id];
	double h = y[1][id];
	double j = y[2][id];
	double d = y[3][id];
	double f = y[4][id];
	double x = y[5][id];
	double Nao = 140.0, Nai = 10.0;
	double Ko = 5.4, Ki = 145.0;
	double PNa_K = 0.01833;
	double E_Na = RTF_mV * log(Nao / Nai);
	double E_K = RTF_mV * log(Ko / Ki);
	double E_K_LR1 = RTF_mV * log((Ko + PNa_K * Nao) / (Ki + PNa_K * Nai));
	double E_si = 7.7 - 13.0287 * log(Cai);
	double am, bm, ah, bh, aj, bj;
	if (fabs(V + 47.13) < 1e-6) {
		am = 3.2;
	} else {
		am = 0.32 * (V + 47.13) / (1.0 - exp(-0.1 * (V + 47.13)));
	}
	bm = 0.08 * exp(-V / 11.0);
	double a = 1.0 - 1.0 / (1.0 + exp(-(V + 40.0) / 0.24));
	ah = a * 0.135 * exp((80.0 + V) / -6.8);
	bh = (1.0 - a) / (0.13 * (1.0 + exp((V + 10.66) / -11.1))) +
	     (a) * (3.56 * exp(0.079 * V) + 3.1e5 * exp(0.35 * V));
	aj = a * (-1.2714e5 * exp(0.2444 * V) - 3.474e-5 * exp(-0.04391 * V)) *
	     (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23)));
	bj = (1.0 - a) * (0.3 * exp(-2.535e-7 * V) / (1.0 + exp(-0.1 * (V + 32.0)))) +
	     (a) * (0.1212 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14))));
	double ad = 0.095 * exp(-0.01 * (V - 5.0)) / (1.0 + exp(-0.072 * (V - 5.0)));
	double bd = 0.07 * exp(-0.017 * (V + 44.0)) / (1.0 + exp(0.05 * (V + 44.0)));
	double af = 0.012 * exp(-0.008 * (V + 28.0)) / (1.0 + exp(0.15 * (V + 28.0)));
	double bf = 0.0065 * exp(-0.02 * (V + 30.0)) / (1.0 + exp(-0.2 * (V + 30.0)));
	double ax = 0.0005 * exp(0.083 * (V + 50.0)) / (1.0 + exp(0.057 * (V + 50.0)));
	double bx = 0.0013 * exp(-0.06 * (V + 20.0)) / (1.0 + exp(-0.04 * (V + 20.0)));
	for (int k = 0; k < 8; k++)
	{
		ydot[k][id] = 0.0;
	}
	ydot[0][id] = am * (1.0 - m) - bm * m;
	ydot[1][id] = ah * (1.0 - h) - bh * h;
	ydot[2][id] = (aj * (1.0 - j) - bj * j);
	ydot[3][id] = (ad * (1.0 - d) - bd * d);
	ydot[4][id] = (af * (1.0 - f) - bf * f);
	ydot[5][id] = ax * (1.0 - x) - bx * x;
	double G_Na = 16.0;
	double I_Na = G_Na * pow(m, 3) * h * j * (V - E_Na);
	double G_si = 0.06;
	double I_si = G_si * d * f * (V - E_si);
	double G_K_array = 0.564;
	double G_K = G_K_array * sqrt(Ko / 5.4);
	double Xi;
	if (V <= -100.0) {
		Xi = 1.0;
	} else if (fabs(V + 77.0) < 1e-6) {
		Xi = (2.837 * 0.04) / exp(0.04 * (-77.0 + 35.0));
	} else {
		Xi = (2.837 * (exp(0.04 * (V + 77.0)) - 1.0)) /
		     ((V + 77.0) * exp(0.04 * (V + 35.0)));
	}
	double I_K = G_K * x * Xi * (V - E_K_LR1);
	double G_K1 = 0.6047 * sqrt(Ko / 5.4);
	double ak1 = 1.02 / (1.0 + exp(0.2385 * (V - E_K - 59.215)));
	double bk1 = (0.49124 * exp(0.08032 * (V - E_K + 5.476)) + exp(0.06175 * (V - E_K - 594.31))) /
	             (1.0 + exp(-0.5143 * (V - E_K + 4.753)));
	double I_K1 = G_K1 * (ak1 / (ak1 + bk1)) * (V - E_K);
	double G_Kp = 0.0183;
	double I_Kp = G_Kp * (V + 87.8789) / (1.0 + exp((7.488 - V) / 5.98));
	double G_b = 0.03921;
	double I_b = G_b * (V + 59.87);
	ydot[6][id] = -1e-4 * I_si + 0.07 * (1e-4 - Cai);
	i_Na_lateral[id] = I_Na;
	i_K1_lateral[id] = I_K1;
	i_comp_lateral[id] = I_si + I_K + I_Kp + I_b;
}

template <unsigned int NMYOS>
__host__ __device__ void myocyte<NMYOS>::rabbit_juncODEfile(int id, int j, double dt)
{
	double Nai = 10.0;
	double Ki = 145.0;
	double ena = (1.0 / FoRT) * log(Nao[j][id] / Nai);
	double ek = (1.0 / FoRT) * log(Ko[j][id] / Ki);

	double GNa = 16.0;
	double inashift = 0.0;
	double am;
	if (fabs(0.1 * (V_myo_junc[j][id] - inashift + 47.13)) < 1e-4)
	{
		am = 0.32 / 0.1;
	}
	else
	{
		am = 0.32 * (V_myo_junc[j][id] - inashift + 47.13) / (1.0 - exp(-0.1 * (V_myo_junc[j][id] - inashift + 47.13)));
	}
	double bm = 0.08 * exp(-(V_myo_junc[j][id] - inashift) / 11.0);
	double V_shifted = V_myo_junc[j][id] - inashift;
	double a = 1.0 - 1.0 / (1.0 + exp(-(V_shifted + 40.0) / 0.24));
	double ah = a * 0.135 * exp((80.0 + V_shifted) / -6.8);
	double bh = (1.0 - a) / (0.13 * (1.0 + exp((V_shifted + 10.66) / -11.1))) +
	            (a) * (3.56 * exp(0.079 * V_shifted) + 3.1e5 * exp(0.35 * V_shifted));
	double aj = a * (-1.2714e5 * exp(0.2444 * V_shifted) - 3.474e-5 * exp(-0.04391 * V_shifted)) *
	            (V_shifted + 37.78) / (1.0 + exp(0.311 * (V_shifted + 79.23)));
	double bj = (1.0 - a) * (0.3 * exp(-2.535e-7 * V_shifted) / (1.0 + exp(-0.1 * (V_shifted + 32.0)))) +
	            (a) * (0.1212 * exp(-0.01052 * V_shifted) / (1.0 + exp(-0.1378 * (V_shifted + 40.14))));
	double taum = 1.0 / (am + bm);
	double tauh = 1.0 / (ah + bh);
	double tauj = 1.0 / (aj + bj);
	d_m_myo_junc[j][id] = (am / (am + bm) - ((am / (am + bm)) - m_myo_junc[j][id]) * exp(-dt / taum) - m_myo_junc[j][id]) / dt;
	d_h_myo_junc[j][id] = (ah / (ah + bh) - ((ah / (ah + bh)) - h_myo_junc[j][id]) * exp(-dt / tauh) - h_myo_junc[j][id]) / dt;
	d_j_myo_junc[j][id] = (aj / (aj + bj) - ((aj / (aj + bj)) - j_myo_junc[j][id]) * exp(-dt / tauj) - j_myo_junc[j][id]) / dt;
	double I_Na = GNa * pow(m_myo_junc[j][id], 3) * h_myo_junc[j][id] * j_myo_junc[j][id] * (V_myo_junc[j][id] - ena);
	double aki = 1.02 / (1 + exp(0.2385 * (V_myo_junc[j][id] - ek - 59.215)));
	double bki = (0.49124 * exp(0.08032 * (V_myo_junc[j][id] + 5.476 - ek)) + exp(0.06175 * (V_myo_junc[j][id] - ek - 594.31))) / (1.0 + exp(-0.5143 * (V_myo_junc[j][id] - ek + 4.753)));
	double kiss = aki / (aki + bki);
	double LR1_IK1_max = 0.6047;
	double I_ki = LR1_IK1_max * sqrt(Ko[j][id] / 5.4) * kiss * (V_myo_junc[j][id] - ek);
	i_Na_myo_cl[j][id] = I_Na;
	i_K1_myo_cl[j][id] = I_ki;
}

template <unsigned int NMYOS>
__host__ __device__ void myocyte<NMYOS>::rabbit_covert_to_Current(int id, double Istim)
{
	double total_membrane_capacitance = C_cell_lat[id];
	for (int j = 0; j < cell_num_neighbors[id]; j++)
	{
		total_membrane_capacitance += NUMofCell_connected[j][id] * C_cell_junc[j][id];
	}
	I_tot_lateral[id] =
		(ryo_f_Ina_lat[id] * i_Na_lateral[id] +
		 ryo_f_k1_lat[id] * i_K1_lateral[id] - Istim) * C_cell_lat[id] +
		i_comp_lateral[id] * total_membrane_capacitance;
	for (int j = 0; j < MAX_CELL_NEIGHBORS; j++)
	{
		I_Na_tot_myo_cl[j][id] = ryo_f_Ina_junc[j][id] * i_Na_myo_cl[j][id] * C_cell_junc[j][id];
		I_K1_tot_myo_cl[j][id] = ryo_f_k1_junc[j][id] * i_K1_myo_cl[j][id] * C_cell_junc[j][id];
		I_tot_myo_cl[j][id] = I_Na_tot_myo_cl[j][id] + I_K1_tot_myo_cl[j][id];
	}
}

template <unsigned int NMYOS>
__host__ __device__ void myocyte<NMYOS>::rabbit_update(int id, double dt)
{
	for (int i = 0; i < 8; i++)
	{
		y[i][id] += ydot[i][id] * dt;
	}
	for (int j = 0; j < cell_num_neighbors[id]; j++)
	{
		m_myo_junc[j][id] += d_m_myo_junc[j][id] * dt;
		h_myo_junc[j][id] += d_h_myo_junc[j][id] * dt;
		j_myo_junc[j][id] += d_j_myo_junc[j][id] * dt;
	}
}
