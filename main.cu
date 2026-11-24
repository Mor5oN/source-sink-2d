#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
using namespace std;
#include "SpatialTissue.cuh"

const double StimStrength = 30.0;
const double StimDur = 2.0;
const bool vertical_gap_only_mode = true;
double dt = 0.001;
const double stabilizationTime = 10.0;
double TotalSimTime = 2000.0;

const unsigned int NUMofMYOS = number_of_myocyte;

#ifndef MYO_ROWS
#define MYO_ROWS 300
#endif
#ifndef MYO_COLS
#define MYO_COLS 300
#endif

int check_simulation_status(const double* V_Cell, int num_cells, double current_time, double stim_end_time)
{
	static bool reached_top = false;
	static bool reached_bottom = false;
	static bool reached_left = false;
	static bool reached_right = false;

	if (current_time <= stim_end_time + 1.0)
	{
		reached_top = false;
		reached_bottom = false;
		reached_left = false;
		reached_right = false;
	}

	if (!reached_top)
	{
		for (int c = 0; c < MYO_COLS; c++)
		{
			if (V_Cell[0 * MYO_COLS + c] > -20.0)
			{
				reached_top = true;
				break;
			}
		}
	}
	if (!reached_bottom)
	{
		for (int c = 0; c < MYO_COLS; c++)
		{
			if (V_Cell[(MYO_ROWS - 1) * MYO_COLS + c] > -20.0)
			{
				reached_bottom = true;
				break;
			}
		}
	}
	if (!reached_left)
	{
		for (int r = 0; r < MYO_ROWS; r++)
		{
			if (V_Cell[r * MYO_COLS + 0] > -20.0)
			{
				reached_left = true;
				break;
			}
		}
	}
	if (!reached_right)
	{
		for (int r = 0; r < MYO_ROWS; r++)
		{
			if (V_Cell[r * MYO_COLS + (MYO_COLS - 1)] > -20.0)
			{
				reached_right = true;
				break;
			}
		}
	}

	if (reached_top && reached_bottom && reached_left && reached_right)
	{
		return 1;
	}

	if (current_time > stim_end_time + 5.0)
	{
		double max_v = -100.0;
		for (int idx = 0; idx < num_cells; idx++)
		{
			if (V_Cell[idx] > max_v)
			{
				max_v = V_Cell[idx];
			}
		}
		if (max_v < -60.0)
		{
			return -1;
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
	cout << "number_of_myocyte: " << number_of_myocyte << endl << endl;
	if (argc < 6)
	{
		fprintf(stderr, "Usage: %s <num_stimulated_cells> <stim_shape> <horizontal_cleft_width_nm> <horizontal_G_gap> <vertical_G_gap> [f_Ina_cleft] [f_Ina_lateral] [TotalSimTime] [lambda]\n", argv[0]);
		return 1;
	}
	std::string num_stimulated_cells_input = argv[1];
	std::string stim_shape_input = argv[2];
	std::string sNUMofMYOS = std::to_string(NUMofMYOS);
	std::string horizontal_width_input = argv[3];
	std::string horizontal_gap_input = argv[4];
	std::string vertical_gap_input = argv[5];
	std::string f_Ina_cleft_input = "0.9";
	std::string f_Ina_lateral_input = "0.1";
	std::string lambda_input = "0.5";

	if (argc >= 7)
	{
		f_Ina_cleft_input = argv[6];
	}
	if (argc >= 8)
	{
		f_Ina_lateral_input = argv[7];
	}
	double f_Ina_cleft_val = std::stod(f_Ina_cleft_input);
	double f_Ina_lateral_val = std::stod(f_Ina_lateral_input);
	if (argc >= 9)
	{
		TotalSimTime = std::stod(argv[8]);
	}
	if (argc >= 10)
	{
		lambda_input = argv[9];
	}
	double lambda_val = std::stod(lambda_input);
	if (lambda_val > 0.5 && lambda_val <= 50.0)
	{
		lambda_val /= 100.0;
	}
	if (lambda_val < 0.0 || lambda_val > 0.5)
	{
		fprintf(stderr, "lambda must satisfy 0 <= value <= 0.5 (or 0 <= percent <= 50).\n");
		return 1;
	}

	double vertical_junc_radius = 0.0;
	double vertical_Na_share = 0.0;
	double vertical_K_share = 0.0;
	SpatialTissue<NUMofMYOS> *wholeTissue =
		new SpatialTissue<NUMofMYOS>(vertical_junc_radius, vertical_Na_share, vertical_K_share, vertical_gap_only_mode, lambda_val);

	int num_stimulated_cells = std::stoi(num_stimulated_cells_input);
	int stim_shape = std::stoi(stim_shape_input);
	wholeTissue -> arrange_center_stimulus_region(num_stimulated_cells, stim_shape);

	double g_cleft_control = 6.0;
	double horizontal_width_of_cleft_myo = std::stof(horizontal_width_input);
	double horizontal_G_gap_myo = std::stof(horizontal_gap_input) * 1e-6;
	double vertical_width_of_cleft_myo = 0.0;
	double vertical_G_gap_myo = std::stof(vertical_gap_input) * 1e-6;

	std::string lambda_suffix = "";
	if (fabs(lambda_val - 0.5) > 1e-6)
	{
		std::ostringstream ss_lambda;
		ss_lambda << "_lambda_" << lambda_val;
		lambda_suffix = ss_lambda.str();
	}
	std::string filename = "center_stim_num_" + num_stimulated_cells_input + "_shape_" + stim_shape_input + "_NofM_" + sNUMofMYOS +
					"_horizontal_width_of_cleft_myo_" + horizontal_width_input +
					"_horizontal_G_gap_myo_" + horizontal_gap_input +
					"_vertical_G_gap_myo_" + vertical_gap_input +
					"_f_Ina_cleft_" + f_Ina_cleft_input +
					"_lateral_" + f_Ina_lateral_input +
					lambda_suffix +
					".txt";

	wholeTissue -> arrange_cells_surface_and_Capacitance(f_Ina_cleft_val, f_Ina_lateral_val);
	wholeTissue -> arrange_cleft_parameter(&g_cleft_control,
										   &horizontal_width_of_cleft_myo,
										   &horizontal_G_gap_myo,
										   &vertical_width_of_cleft_myo,
										   &vertical_G_gap_myo);
	wholeTissue -> build_sparse_matrix();
	wholeTissue -> cpu_copy_gpu();

	FILE *outputFile = fopen(filename.c_str(), "w");
	if (outputFile == nullptr)
	{
		fprintf(stderr, "Unable to open output file: %s\n", filename.c_str());
		delete wholeTissue;
		return 1;
	}
	fprintf(outputFile, "t ");
	for (int idx = 0; idx < NUMofMYOS; idx++)
	{
		fprintf(outputFile, "myo_%d ", idx + 1);
	}
	fprintf(outputFile, "\n");

	struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);

	double time = 0.0;
	int total_steps = static_cast<int>(TotalSimTime / dt);
	int last_checked_status = 0;
	for (int i = 0; i < total_steps; i++)
	{
		time = i * dt;

		bool checked_status = false;
		int status = 0;
		if (i > 0 && i % 1000 == 0)
		{
			wholeTissue -> copy_voltage_to_host();
			checked_status = true;
			double stim_end_time = stabilizationTime + StimDur;
			status = check_simulation_status(wholeTissue -> cable_grid -> V_Cell, NUMofMYOS, time, stim_end_time);
			last_checked_status = status;
		}

		bool output_time_check = (i == 0) || (i > 0 && i % 10000 == 0);
		if (output_time_check && !checked_status)
		{
			wholeTissue -> copy_voltage_to_host();
		}

		if (output_time_check || status != 0)
		{
			fprintf(outputFile, "%f", time);
			for (int idx = 0; idx < NUMofMYOS; idx++)
			{
				fprintf(outputFile, " %f", wholeTissue -> cable_grid -> V_Cell[idx]);
			}
			fprintf(outputFile, "\n");
		}

		if (status == 1)
		{
			printf("Conduction reached the boundary at time = %f ms. Stopping simulation early.\n", time);
			break;
		}
		else if (status == -1)
		{
			printf("Action potential died out (max V < -60 mV) at time = %f ms. Stopping simulation early.\n", time);
			break;
		}

		double Istim = 0.0;
		int stimulus_id = 0;
		if (time >= stabilizationTime && time < stabilizationTime + StimDur)
		{
			Istim = StimStrength;
			stimulus_id = 1;
		}
		wholeTissue -> step(dt, Istim, time, stimulus_id);
	}

	wholeTissue -> copy_voltage_to_host();
	double stim_end_time = stabilizationTime + StimDur;
	int final_check_status = check_simulation_status(wholeTissue -> cable_grid -> V_Cell, NUMofMYOS, time, stim_end_time);

	int final_status = 0;
	if (last_checked_status == 1 || final_check_status == 1)
	{
		final_status = 1;
	}
	else if (last_checked_status == -1 || final_check_status == -1)
	{
		final_status = -1;
	}
	else
	{
		double max_v = -100.0;
		for (int idx = 0; idx < NUMofMYOS; idx++)
		{
			if (wholeTissue -> cable_grid -> V_Cell[idx] > max_v)
			{
				max_v = wholeTissue -> cable_grid -> V_Cell[idx];
			}
		}
		if (max_v >= -60.0)
		{
			final_status = 2;
		}
		else
		{
			final_status = -1;
		}
	}

	if (final_status == 1)
	{
		fprintf(outputFile, "#STATUS SUCCESS\n");
	}
	else if (final_status == 2)
	{
		fprintf(outputFile, "#STATUS INSUFFICIENT\n");
	}
	else
	{
		fprintf(outputFile, "#STATUS FAILED\n");
	}

	gettimeofday(&endTime, NULL);
	printf("Gpu use time: %ld s\n", (endTime.tv_sec - startTime.tv_sec));
	fclose(outputFile);
	delete wholeTissue;
}
