"""
Extract the input data: BTD, surf_temp, surf_pressure, vapor_profile, four angles
keep the neccessary data: lat, lon, cld
Normalize the input data 
"""
import os
import glob
import numpy as np
from Tools import iasi, era5_rttov, irtt_config
import netCDF4 as nc
from multiprocessing import Pool


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


def initNCfile(filename):
    if os.path.exists(f'{OUTPUT_MODEL}/{filename}'):
        print(f"{filename} already exists. Delete it!")
        os.remove(f'{OUTPUT_MODEL}/{filename}')

    # 创建目标 NetCDF 文件
    filename = f'{OUTPUT_MODEL}/{filename}' 
    print("Initializing ", filename)
    output_file = nc.Dataset(filename, 'w', format='NETCDF4')
    # 创建一个可扩展的维度，用于存储点的数量
    output_file.createDimension('dim1', None)
    output_file.createDimension('dim2', None)
    output_file.createDimension('dim3', None)

    # 创建变量用于存储符合条件的点的经纬度及其他属性
    output_file.createVariable('lat', 'f4', ('dim1', 'dim2'))
    output_file.createVariable('lon', 'f4', ('dim1', 'dim2'))
    output_file.createVariable('cld', 'f4', ('dim1', 'dim2'))

    output_file.createVariable('input_data', 'f4', ('dim1', 'dim2', 'dim3'))
    output_file.createVariable('output_data', 'f4', ('dim1', 'dim2'))
    output_file.createVariable('column', 'f4', ('dim1', 'dim2'))
    output_file.createVariable('uncertainty', 'f4', ('dim1', 'dim2'))

    # 关闭文件 (后面会打开文件追加数据)
    output_file.close()


def writeNCfile(filename, level1_file, params):
    # Read the orbit
    orbit = f"{level1_file.split('_')[4]}_{level1_file.split('_')[5]}"
    print(f"Processing orbit {orbit}...")
    date = orbit[:8]

    # Read ERA5 data by the day
    era5_data = era5_rttov()
    era5_data.readcoef(date, profile=False)

    try:
        # 判断轨道对应的一级数据是否存在，若不存在则跳过
        sate = iasi('Metop-B')
        sate.path1level = level1_file
        sate.read1Level()
    except Exception as error:
        print(f"Reading LEVEL1 file Error: {error}")
        return None
    try:
        level2_file = glob.glob(f"/exports/d4/hhr346/Metop-B/LEVEL2/*{orbit}*")[0]
        sate.path2level = level2_file
        sate.read2Level()
    except Exception as error:
        print(f"Reading LEVEL2 file Error: {error}")
        return None

    # Append the data to the lists in a vectorized manner
    surf_temp = era5_data.interpolate(sate.unixtime, sate.lat, sate.lon, era5_data.skinTemp)
    surf_pressure = era5_data.interpolate(sate.unixtime, sate.lat, sate.lon, era5_data.surfPressure)
    start_idx = np.where(sate.wavenumber == params.wave_num_s)[0][0]
    end_idx = np.where(sate.wavenumber == params.wave_num_e)[0][0] + 1
    radiance = sate.radiance[:, :, start_idx:end_idx]
    radiance = brightness_temperature(radiance, sate.wavenumber[start_idx:end_idx])
    angles = np.concatenate((sate.zenith[:, :, np.newaxis], sate.solar_zenith[:, :, np.newaxis], sate.azimuth[:, :, np.newaxis], sate.solar_azimuth[:, :, np.newaxis]), axis=2)

    # input_varible = np.concatenate((radiance[:, :, :], surf_temp[:, :, np.newaxis], surf_pressure[:, :, np.newaxis]), axis=2)
    input_varible = np.concatenate((radiance[:, :, :], surf_pressure[:, :, np.newaxis], angles), axis=2)
    print(f"The shape of the input variable is {np.shape(input_varible)}")


    # 打开目标文件（以追加模式打开）
    print(f"Appending data to {filename}...")
    output_file = nc.Dataset(f'{OUTPUT_MODEL}/{filename}', 'a')

    lat_var = output_file.variables['lat']
    lon_var = output_file.variables['lon']
    cld_var = output_file.variables['cld']
    input_data_var = output_file.variables['input_data']

    # Write into the NetCDF file
    lat_var[:] = sate.lat
    lon_var[:] = sate.lon
    cld_var[:] = sate.cld
    input_data_var[:] = input_varible
    output_file.close()


if __name__ == "__main__":
    import datetime
    params = irtt_config('../params.json')
    params.begin = datetime.date(2022, 1, 1)
    params.end = datetime.date(2022, 12, 31)
    OUTPUT_MODEL = '/your_output_eval_path/MODEL/'

    delta = datetime.timedelta(days=1)
    d = params.begin

    def process_day(func_params):
        time_target, level1_file, params = func_params
        print(f"Processing {time_target}...")
        try:
            filename = os.path.basename(level1_file)
            filename = f"{params.gas_name}_{filename}"
            initNCfile(filename)
            writeNCfile(filename, level1_file, params)
        except Exception as error:
            print('Processing error, ', error)

    # Multi-processing version
    while d <= params.end:
        func_params = []
        time_target = d.strftime("%Y%m%d")
        level1_files = glob.glob(f"{params.level1_path}/IASI_xxx_1C_M01_{time_target}*")
        for level1_file in level1_files:
            func_params.append((time_target, level1_file, params))

        with Pool(processes=8) as pool:
            pool.map(process_day, func_params)
        d += delta
