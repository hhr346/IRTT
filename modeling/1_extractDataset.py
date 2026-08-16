"""
Extract dataset from IASI Level2 data and ERA5 data, and save it into a NetCDF file.
"""
import glob
import numpy as np
from Tools import iasi, irtt_config, era5_rttov
import netCDF4
from multiprocessing import Pool
import numpy as np


def initNCfile(filename):
    # 创建目标 NetCDF 文件
    filename = f'{OUTPUT}/dataset_extract/{filename}.nc' 
    print("Initializing ", filename)
    output_file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
    # 创建一个可扩展的维度，用于存储点的数量
    output_file.createDimension('dim_point', None)
    output_file.createDimension('dim_radiance', None)
    output_file.createDimension('dim_profile', None)

    # 创建变量用于存储符合条件的点的经纬度及其他属性
    output_file.createVariable('lat', 'f4', ('dim_point',))
    output_file.createVariable('lon', 'f4', ('dim_point',))

    output_file.createVariable('time', str, ('dim_point',))
    output_file.createVariable('zenith', 'f4', ('dim_point',))
    output_file.createVariable('azimuth', 'f4', ('dim_point',))
    output_file.createVariable('solar_zenith', 'f4', ('dim_point',))
    output_file.createVariable('solar_azimuth', 'f4', ('dim_point',))

    output_file.createVariable('radiance', 'f4', ('dim_point', 'dim_radiance'))
    output_file.createVariable('pressure_profile', 'f4', ('dim_profile', ))         # Only one constant profile!
    output_file.createVariable('temp_profile', 'f4', ('dim_point', 'dim_profile'))
    output_file.createVariable('vapor_profile', 'f4', ('dim_point', 'dim_profile'))

    output_file.createVariable('surf_z', 'f4', ('dim_point',))
    output_file.createVariable('surf_pressure', 'f4', ('dim_point',))
    output_file.createVariable('surf_temp', 'f4', ('dim_point',))

    # ERA5 data
    output_file.createVariable('surf_type', 'f4', ('dim_point',))
    output_file.createVariable('skin_temperature', 'f4', ('dim_point',))
    output_file.createVariable('2m_temperature', 'f4', ('dim_point',))
    output_file.createVariable('10m_u_wind', 'f4', ('dim_point',))
    output_file.createVariable('10m_v_wind', 'f4', ('dim_point',))
    
    # output_file.createVariable('surf_emissivity', 'f4', ('dim_point',))
    # output_file.createVariable('h2o_column', 'f4', ('dim_point',))

    # 关闭文件 (后面会打开文件追加数据)
    output_file.close()


def writeNCfile(level2_file, params):
    # Read the orbit
    orbit = f"{level2_file.split('_')[4]}_{level2_file.split('_')[5]}"
    print(f"Processing orbit {orbit}...")

    # 打开目标文件（以追加模式打开）
    output_file = netCDF4.Dataset(f'{OUTPUT}/dataset_extract/dataset_{orbit}.nc', 'a')
    # 读取ERA5数据
    date = orbit[:8]
    print(f"Processing {date}...")
    era5_data = era5_rttov()
    era5_data.readcoef(date)

    # 准备要追加的变量
    lat_var = output_file.variables['lat']
    lon_var = output_file.variables['lon']
    time_var = output_file.variables['time']
    solar_zenith_var = output_file.variables['solar_zenith']
    solar_azimuth_var = output_file.variables['solar_azimuth']
    zenith_var = output_file.variables['zenith']
    azimuth_var = output_file.variables['azimuth']

    radiance_var = output_file.variables['radiance']
    
    # Profile data, either from ERA5 or LEVEL2
    temperature_var = output_file.variables['temp_profile']
    vapor_var = output_file.variables['vapor_profile']
    pressure_var = output_file.variables['pressure_profile']

    surf_z_var = output_file.variables['surf_z']
    surf_pressure_var = output_file.variables['surf_pressure']

    surf_temp_var = output_file.variables['surf_temp']
    # surf_emissivity_var = output_file.variables['surf_emissivity']
    # h2o_column_var = output_file.variables['h2o_column']

    # ERA5 data
    surf_type_var = output_file.variables['surf_type']
    skin_temperature_var = output_file.variables['skin_temperature']
    s2m_temperature_var = output_file.variables['2m_temperature']
    s10m_u_wind_var = output_file.variables['10m_u_wind']
    s10m_v_wind_var = output_file.variables['10m_v_wind']


    # Initialize lists to store the data
    lat_list = []
    lon_list = []
    time_list = []
    solar_zenith_list = []
    solar_azimuth_list = []
    zenith_list = []
    azimuth_list = []

    radiance_list = []
    temperature_list = []
    vapor_list = []
    pressure_list = []

    surf_z_list = []
    surf_pressure_list = []

    surf_temp_list = []
    # surf_emissivity_list = []
    # h2o_column_list = []

    # ERA5 data
    surf_type_list = []
    skin_temperature_list = []
    s2m_temperature_list = []
    s10m_u_wind_list = []
    s10m_v_wind_list = []

    try:
        # 判断轨道对应的一级数据是否存在，若不存在则跳过
        level1_file = glob.glob(f"{params.level1_path}*{orbit}*")[0]
        sate = iasi('Metop-B')
        sate.path1level = level1_file
        sate.read1Level()
        sate.path2level = level2_file
        sate.read2Level()

        # 截取需要的radiance
        start_idx = np.where(sate.wavenumber == params.wave_num_s)[0][0]
        end_idx = np.where(sate.wavenumber == params.wave_num_e)[0][0] + 1
        sate.radiance = sate.radiance[:, :, start_idx:end_idx]

        # 不再分纬度，而是重新随机选择
        SAMPLE = 300
        new_mask = np.zeros_like(sate.lat, dtype=bool)

        mask = ((sate.cld == 1.0)|(sate.cld == 2.0))
        true_indices = np.argwhere(mask)

        if len(true_indices) > SAMPLE:
            selected_indices = np.random.choice(len(true_indices), size=SAMPLE, replace=False)
            new_mask[tuple(true_indices[selected_indices].T)] = True
        else:
            new_mask[tuple(true_indices.T)] = True
        print(f"Choose {np.sum(new_mask)} points in orbit {orbit}.")


        # Append the data to the lists in a vectorized manner
        lat_list.extend(sate.lat[new_mask].tolist())
        lon_list.extend(sate.lon[new_mask].tolist())
        time_list.extend(sate.time[new_mask].tolist())
        zenith_list.extend(sate.zenith[new_mask].tolist())
        azimuth_list.extend(sate.azimuth[new_mask].tolist())
        solar_zenith_list.extend(sate.solar_zenith[new_mask].tolist())
        solar_azimuth_list.extend(sate.solar_azimuth[new_mask].tolist())
        radiance_list.extend(sate.radiance[new_mask].tolist())

        # ERA5 data interpolation
        time_mask = sate.unixtime[new_mask]
        lat_mask = sate.lat[new_mask]
        lon_mask = sate.lon[new_mask]

        skin_temperature_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.skinTemp).tolist())
        s2m_temperature_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.s2mTemp).tolist())
        surf_pressure_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.surfPressure).tolist())

        # surf_temp_list.extend(sate.surface_temp[new_mask].tolist())
        # skin_temperature_list.extend(sate.surface_temp[new_mask].tolist())
        # s2m_temperature_list.extend(sate.surface_temp[new_mask].tolist())
        # surf_pressure_list.extend(sate.surface_pressure[new_mask].tolist())

        s10m_u_wind_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.s10mUwind).tolist())
        s10m_v_wind_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.s10mVwind).tolist())

        interp_surf_type = era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.surfaceType)
        interp_surf_type = np.where(interp_surf_type > 0, 1, 0)         # 防止海陆边缘的点被取分数
        surf_type_list.extend(interp_surf_type.tolist())
        surf_z_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.surfaceZ).tolist())

        # Profile data 
        temperature_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.temp_p).tolist())
        vapor_list.extend(era5_data.interpolate(time_mask, lat_mask, lon_mask, era5_data.vapor_p).tolist())
        if len(pressure_list) == 0:
            pressure_list.extend(era5_data.pressure_p.tolist())

        # Write into the NetCDF file
        lat_var[:] = np.array(lat_list)
        lon_var[:] = np.array(lon_list)
        time_var[:] = np.array(time_list)
        zenith_var[:] = np.array(zenith_list)
        azimuth_var[:] = np.array(azimuth_list)
        solar_zenith_var[:] = np.array(solar_zenith_list)
        solar_azimuth_var[:] = np.array(solar_azimuth_list)
        radiance_var[:] = np.array(radiance_list)
        print(f"Shape of radiance is {np.shape(radiance_var)}")

        temperature_var[:] = np.array(temperature_list)
        vapor_var[:] = np.array(vapor_list)
        pressure_var[:] = np.array(pressure_list)
        surf_temp_var[:] = np.array(surf_temp_list)
        # surf_emissivity_var[:] = np.array(surf_emissivity_list)
        # h2o_column_var[:] = np.array(h2o_column_list)

        # ERA5 data
        surf_z_var[:] = np.array(surf_z_list)
        surf_pressure_var[:] = np.array(surf_pressure_list)
        surf_type_var[:] = np.array(surf_type_list)
        skin_temperature_var[:] = np.array(skin_temperature_list)
        s2m_temperature_var[:] = np.array(s2m_temperature_list)
        s10m_u_wind_var[:] = np.array(s10m_u_wind_list)
        s10m_v_wind_var[:] = np.array(s10m_v_wind_list)
        print("Writing data to NetCDF file done.")
        output_file.close()

    except Exception as error:
        print(f"In orbit {orbit}, \033[0;31m{error}\033[0m")
        return None



if __name__ == "__main__":
    import datetime
    import os

    params = irtt_config('../params.json')
    OUTPUT = os.path.join(params.output_folder, params.gas_name)
    os.makedirs(f"{OUTPUT}/", exist_ok=True)
    os.makedirs(f"{OUTPUT}/dataset_extract/", exist_ok=True)
    delta = datetime.timedelta(days=1)
    d = params.begin

    def process_day(func_param):
        time_target, level2_file, params = func_param
        print(f"Processing {time_target}...")
        # try:
        orbit = f"{level2_file.split('_')[4]}_{level2_file.split('_')[5]}"
        initNCfile(f'dataset_{orbit}')
        writeNCfile(level2_file, params)

    while d <= params.end:
        func_params = []
        time_target = d.strftime("%Y%m%d")
        level2_files = glob.glob(f"{params.level2_path}*IASI_SND_02_M01_{time_target}*")
        for level2_file in level2_files:
            func_params.append((time_target, level2_file, params))
        d += delta

        with Pool(processes=16) as pool:
            pool.map(process_day, func_params)
