"""
Calculate the column concentration of the target gas based on the simulation data and the extracted data from ERA5.
"""
import datetime
import glob
import os
import numpy as np
import netCDF4 as nc
from multiprocessing import Pool
from Tools import irtt_config

class extractData():
    filepath = None
    def __init__(self) -> None:
        pass
    
    def readData(self):
        try:
            print('\033[0;33mOpening file %s\033[0m' %self.filepath)
            ncfile = nc.Dataset(self.filepath, mode='r')
            self.surf_temp = np.asarray(ncfile.variables['skin_temperature'][:])
            self.surf_pressure = np.asarray(ncfile.variables['surf_pressure'][:])
            self.pressure_profile = np.asarray(ncfile.variables['pressure_profile'][0, :])
        except Exception as error:
            print('Reading extractData Error', error)


class simData():
    filepath = None
    def __init__(self) -> None:
        self.label = None
        self.wavenum = np.arange(params.wave_num_s, params.wave_num_e + 0.25, 0.25)

    def readData(self):
        try:
            print('\033[0;33mOpening file %s\033[0m' %self.filepath)
            ncfile = nc.Dataset(self.filepath, mode='r')
            self.profile = np.asarray(ncfile.variables['profile'][:, :])
            self.radiance = np.asarray(ncfile.variables['radiance'][:, :])
            self.simulation = np.asarray(ncfile.variables['simulation'][:, :])
            ncfile.close()

        except Exception as error:
            print('Reading simData Error', error)
    
    def calColumn(self, p_profile, surf_p):
        """
        Calculate the column concentration of the target gas
        """
        g = 9.81  # m/s^2
        N_A = 6.022e23 # molec/mole
        M_air = 0.02897 # kg/mol

        # 根据地表压力截取有效的气压剖面
        surf_pressure = surf_p[:, np.newaxis] * np.ones((1, np.shape(p_profile)[0]))
        p_profile = p_profile[np.newaxis, :] * np.ones((np.shape(surf_p)[0], 1))
        p_profile = np.minimum(p_profile, surf_pressure)  # 取最小值

        # 计算层间压差 Δp
        delta_p = np.diff(p_profile, axis=1)
        # 计算气体柱浓度
        profile_mole = (self.profile[:, :-1] + self.profile[:, 1:]) / 2e6  # ppm 转为无量纲
        column_density_layers = 100 * delta_p / g * profile_mole / M_air * N_A / 1e4 # molecules/cm^2

        total_column = np.sum(column_density_layers, axis=1)
        self.total_column = total_column.squeeze()


if __name__ == "__main__":
    params = irtt_config('../params.json')
    day = params.begin
    delta = datetime.timedelta(days=1)
    OUTPUT_BAK = params.output_folder + f"{params.gas_name}/dataset_extract/raw/"
    OUTPUT_OUT = params.output_folder + f"{params.gas_name}/dataset_simulate/raw/"
    os.makedirs(OUTPUT_BAK, exist_ok=True)
    os.makedirs(OUTPUT_OUT, exist_ok=True)
    MULTI_TIMES = params.multi_times

    wavenum = np.arange(params.wave_num_s, params.wave_num_e + 0.25, 0.25)
    print(f"The calculated wavenumber range is {wavenum}\n")
    wavelen = 1.0e7 / wavenum[::-1]     # cm-1 to nm
    wavelen_length = np.shape(wavelen)[0]


    # Read all the simulation files
    simulation_files = sorted(glob.glob(f'{OUTPUT_OUT}/split_*.nc' ))
    background_files = sorted(glob.glob(f'{OUTPUT_BAK}/split_*.nc' ))
    func_params = [(simulation_file, background_file) for simulation_file, background_file in zip(simulation_files, background_files)]

    def process_day(simulation_file, background_file):
        extraction = extractData()
        extraction.filepath = background_file
        extraction.readData()

        simulation = simData()
        simulation.filepath = simulation_file
        simulation.readData()

        # Transform the radiance into Brightness 

        points_num = np.shape(simulation.radiance)[0]
        print(f"The number of points in this file is {points_num}")
        simulation.calColumn(extraction.pressure_profile, np.repeat(extraction.surf_pressure, MULTI_TIMES))

        # Write into the simulation file
        with nc.Dataset(simulation_file, 'a') as output_file:
            try:
                output_file.createVariable('total_column', 'f4', ('dim_point',))
            except Exception:
                pass

            total_column_var = output_file.variables['total_column']
            total_column_var[:] = simulation.total_column
            print(f"Shape of the total column is {np.shape(simulation.total_column)}")

    with Pool(processes=24) as pool:
        pool.starmap(process_day, func_params)
