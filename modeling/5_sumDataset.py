"""
Extract everything you need for the dataset into a single file
"""
import datetime
import glob
import os
import numpy as np
import netCDF4 as nc
from Tools import irtt_config

def initNCfile(filename):
    # 创建目标 NetCDF 文件
    filepath = f'{OUTPUT_OUT}/{filename}.nc' 
    print("Initializing ", filepath)
    output_file = nc.Dataset(filepath, 'w', format='NETCDF4')

    # 创建一个可扩展的维度，用于存储点的数量
    output_file.createDimension('dim_point', None)
    output_file.createDimension('dim_input', None)

    # 创建变量
    output_file.createVariable('input', 'f4', ('dim_point', 'dim_input'))
    output_file.createVariable('output', 'f4', ('dim_point', ))
    output_file.createVariable('lat', 'f4', ('dim_point', ))
    output_file.createVariable('lon', 'f4', ('dim_point', ))
    output_file.close()


class extractData():
    filepath = None
    def __init__(self) -> None:
        pass
    
    def readData(self):
        try:
            print('\033[0;33mOpening file %s\033[0m' %self.filepath)
            ncfile = nc.Dataset(self.filepath, mode='r')
 
            t2m = np.asarray(ncfile.variables['2m_temperature'][:])
            self.lat = np.asarray(ncfile.variables['lat'][:])
            self.lon = np.asarray(ncfile.variables['lon'][:])
            self.skin_temperature = np.asarray(ncfile.variables['skin_temperature'][:])
            self.thermal_contrast = self.skin_temperature - t2m
            self.surf_pressure = np.asarray(ncfile.variables['surf_pressure'][:])
            self.emis = np.asarray(ncfile.variables['emis_uw'][:])
            self.zenith = np.asarray(ncfile.variables['zenith'][:])
            self.solar_zenith = np.asarray(ncfile.variables['solar_zenith'][:])
            self.azimuth = np.asarray(ncfile.variables['azimuth'][:])
            self.solar_azimuth = np.asarray(ncfile.variables['solar_azimuth'][:])
            self.temp_profile = np.asarray(ncfile.variables['temp_profile'][:])
            self.angles = np.concatenate((self.zenith[:, np.newaxis], self.solar_zenith[:, np.newaxis], self.azimuth[:, np.newaxis], self.solar_azimuth[:, np.newaxis]), axis=1)
            ncfile.close()

        except Exception as error:
            print('Reading extractData Error', error)

def wien_invert(L, nu):
    """
    反演亮温（单位：K），输入波数单位为 cm⁻¹，辐射单位为 mW/(m²·sr·cm⁻¹)。
    参数:
        L (float): 辐射强度 [mW/(m²·sr·m⁻¹)]
        nu_cm (float): 波数 [cm⁻¹]
    返回:
        float: 亮温 [K]
    """
    C1 = 1.19104E-5
    C2 = 1.43877
    T = (nu*C2)/np.log(C1*nu**3/L + 1.0)  # 单位 K
    return T


def brightness_temperature(radiance, wavenumber):
    """计算 IASI 亮温（向量化）"""
    T_B = np.zeros_like(radiance)
    for i in range(len(radiance)):
        T_B[i] = wien_invert(radiance[i], wavenumber)
    return T_B


class simData():
    filepath = None
    def __init__(self) -> None:
        pass
    
    def readData(self):
        try:
            print('\033[0;33mOpening file %s\033[0m' %self.filepath)
            ncfile = nc.Dataset(self.filepath, mode='r')
 
            self.radiance = np.asarray(ncfile.variables['simulation'][:])
            self.wavenumber = np.asarray(ncfile.variables['wavenum'][:])
            self.radiance = brightness_temperature(self.radiance, self.wavenumber)
            self.column = np.asarray(ncfile.variables['total_column'][:])
            ncfile.close()

        except Exception as error:
            print('Reading simData Error', error)


def normalize(data, min, max, scale=1, log=False):
    data = data * scale
    if log:
        data = np.log(data)
    data = (data - min) / (max - min)
    data = data * 2 - 1
    return data

def denormalize(data, min, max, scale=1, log=False):
    data = (data + 1) / 2
    data = data * (max - min) + min 
    if log:
        data = np.exp(data)
    data = data / scale
    return data


if __name__ == "__main__":
    params = irtt_config('../params.json')
    day = params.begin
    delta = datetime.timedelta(days=1)
    OUTPUT_BAK = params.output_folder + f"{params.gas_name}/dataset_extract/raw/"
    OUTPUT_OUT = params.output_folder + f"{params.gas_name}/dataset_simulate/raw/"
    MULTI_TIMES = params.multi_times

    input_total = []
    output_total = []
    lat_total = []
    lon_total = []
    surf_type_total = []

    # Read all the simulation files
    simulation_files = sorted(glob.glob(f'{OUTPUT_OUT}/split_*.nc'))
    total_num = 0
    for simulation_file in simulation_files:
        try:
            # Read the simulation data
            simulation = simData()
            simulation.filepath = simulation_file
            simulation.readData()
            print(f"The number of points in this file is {np.shape(simulation.column)[0]}")

            # Read the background extracted data
            filename = os.path.basename(simulation_file)
            background_file = glob.glob(f'{OUTPUT_BAK}/{filename}')[0]

            background = extractData()
            background.filepath = background_file
            background.readData()

            index_all = np.arange(np.shape(background.lat)[0])
            # Index for the simulation data
            start_points = index_all * MULTI_TIMES
            expanded = np.array([np.arange(start, start + MULTI_TIMES) for start in start_points])
            index_all_simulate = expanded.flatten()
            # Index for the background data, repeated MULTI_TIMES times
            index_all_background = np.repeat(index_all, MULTI_TIMES)
            total_num += np.shape(index_all_simulate)[0]

            # Write some auxilary information
            lat = background.lat[index_all_background]
            lon = background.lon[index_all_background]

            radiance = simulation.radiance[index_all_simulate]
            column = simulation.column[index_all_simulate]
            surf_temp = background.skin_temperature[index_all_background]
            surf_pressure = background.surf_pressure[index_all_background]
            tc = background.thermal_contrast[index_all_background]
            emis = background.emis[index_all_background]
            angles = background.angles[index_all_simulate]
            temp_profile = background.temp_profile[index_all_background]

            input_varible = np.concatenate((radiance[:, :], surf_temp[:, np.newaxis], surf_pressure[:, np.newaxis], emis[:, np.newaxis], angles, temp_profile), axis=1)
            print(f"The shape of the input variable is {np.shape(input_varible)}")
            input_total.extend(input_varible)
            output_total.extend(column)

            lat_total.extend(lat)
            lon_total.extend(lon)

        except Exception as error:
            print('Writing simData Error!', error)

    # Sift out the infinite and NaN values
    input_idx_nan = np.isnan(input_total).any(axis=1)
    output_idx_nan = np.isnan(output_total)
    input_idx_inf = np.isinf(input_total).any(axis=1)
    output_idx_inf = np.isinf(output_total)
    total_idx = input_idx_nan | output_idx_nan | input_idx_inf | output_idx_inf

    # print(np.argwhere(input_idx_nan)[:10])
    # row_nan = np.where(input_idx_nan)[0][:10]
    # print(row_nan)
    # print(np.array(input_total)[row_nan])

    print(f"The number of input NaN and Inf values is {np.sum(input_idx_nan)} and {np.sum(input_idx_inf)}")
    print(f"The number of output NaN and Inf values is {np.sum(output_idx_nan)} and {np.sum(output_idx_inf)}")
    print(f"The number of NaN and Inf values is {np.sum(total_idx)}")

    input_total = np.asarray(input_total)[~total_idx]
    output_total = np.asarray(output_total)[~total_idx]
    print(f"Min of output_total is {np.min(output_total)}, max is {np.max(output_total)}")

    lat_total = np.asarray(lat_total)[~total_idx]
    lon_total = np.asarray(lon_total)[~total_idx]

    # Initialize the total file
    filename = 'total_dataset_full'
    filepath = f'{OUTPUT_OUT}/{filename}.nc' 
    initNCfile(filename)
    # Write into the total file
    print("Writing ", filepath)
    print("The total number of points is ", total_num)

    with nc.Dataset(filepath, 'a') as output_file:
        input_var = output_file.variables['input']
        input_var[:] = input_total

        output_var = output_file.variables['output']
        output_var[:] = output_total
        lat_var = output_file.variables['lat']
        lat_var[:] = lat_total
        lon_var = output_file.variables['lon']
        lon_var[:] = lon_total
