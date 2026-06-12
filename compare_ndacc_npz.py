"""
Compare the IRTT with the IASI Products and NDACC
"""
from multiprocessing import Pool
import glob
import xarray as xr
import netCDF4
import numpy as np
import datetime
from pyhdf.SD import SD, SDC
import pandas as pd
import numpy as np
from Tools import era5_rttov2
import glob

def kg_per_m2_to_molecules_per_cm2(mass_kg_per_m2):
    molar_mass_co_kg_per_mol = 0.02801  # CO 的摩尔质量 (kg/mol)
    avogadro_number = 6.022e23          # 阿伏伽德罗常数 (分子/mol)
    
    molecules_per_cm2 = (mass_kg_per_m2 / molar_mass_co_kg_per_mol) * avogadro_number * 1E-4
    return molecules_per_cm2


class satellite():
    def __init__(self):
        pass

    def mjd2k_to_datetime(self, mjd2k):
        base_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
        result = base_date + datetime.timedelta(seconds=mjd2k)
        return result
 
    def read_model(self, file_path):
        # print('\033[0;33mOpening model file %s\033[0m' %file_path)
        ds = xr.open_dataset(file_path)
        self.model_co = ds['column'].values[:, :]
        self.uncertainty = ds['uncertainty'].values[:, :]
        self.cld = ds['cld'].values[:, :]

        self.lat = ds['lat'].values[:, :]
        self.lon = ds['lon'].values[:, :]
    
    def read_product(self, file_path):
        # print('\033[0;33mOpening l2 file %s\033[0m' %file_path)
        ncfile = netCDF4.Dataset(file_path, mode='r')
        self.co = np.asarray(ncfile.variables['integrated_co'][:, :])
        self.co = kg_per_m2_to_molecules_per_cm2(self.co)  # Convert to molecules/cm2
        self.cld = np.asarray(ncfile.variables['flag_cldnes'][:, :])   # 1-4, 1 and 2 can be seen as clear
        self.quality = np.asarray(ncfile.variables['co_qflag'][:, :])   # Quality flag

        self.lat = np.asarray(ncfile.variables['lat'][:, :])
        self.lon = np.asarray(ncfile.variables['lon'][:, :])
        self.time = np.asarray(ncfile.variables['record_start_time'][:]) # seconds since 2000-1-1 00:00
        self.time = np.tile(self.time[:, np.newaxis], (1, self.co.shape[1]))
        ncfile.close()


class ndaccSite:
    def __init__(self, name):
        self.name = name

    def mjd2k_to_datetime(self, mjd2k):
        base_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
        result = base_date + datetime.timedelta(seconds=mjd2k)
        return result
    
    def mjd2k_to_seconds(self, mjd2k):
        return mjd2k*24*60*60

    def read_file(self, file_paths):
        file_paths = glob.glob(file_paths)
        for i, file_path in enumerate(file_paths):
            if i == 0:
                # print(f"Opening file {file_path}")

                hdf_file = SD(file_path, SDC.READ)
                self.lat = hdf_file.select('LATITUDE.INSTRUMENT')[0]
                self.lon = hdf_file.select('LONGITUDE.INSTRUMENT')[0]
                self.time = hdf_file.select('DATETIME')[:]
                self.time = self.mjd2k_to_seconds(self.time)        # Convert to seconds from 2000 00:00
                self.datetime = np.array([self.mjd2k_to_datetime(t) for t in self.time])          # Convert to the datetime format
                self.column = hdf_file.select('CO.COLUMN_ABSORPTION.SOLAR')[:]
            else:
                time = hdf_file.select('DATETIME')[:]
                time = self.mjd2k_to_seconds(time)        # Convert to seconds from 2000 00:00
                datetime = np.array([self.mjd2k_to_datetime(t) for t in time])          # Convert to the datetime format
                column = hdf_file.select('CO.COLUMN_ABSORPTION.SOLAR')[:]

                self.time = np.hstack((self.time, time))
                self.datetime = np.hstack((self.datetime, datetime))
                self.column = np.hstack((self.column, column))
 

def siftPoint(time_target, product_files, model_files, ndacc):
    sate = satellite()
    product_output_list = []
    ndacc_product_list = []

    model_output_list = []
    ndacc_model_list = []

    # Read from the product
    for product_file in product_files:
        filename_split = product_file.split('_')
        orbit = ('_').join(filename_split[4:6])
        # print(f"Processing the product data {orbit}")

        try:
            sate.read_product(product_file)
            product_output, ndacc_product = compare(time_target, sate, ndacc)
        except Exception as e:
            print(f"Error reading product file: {e}")
            continue

        # The same orbit processing
        if len(model_files) == 0:
            print("Nothing on the model output.")
            continue

        sate_mask = np.char.find(model_files, orbit) != -1
        model_files = np.array(model_files)
        model_file = model_files[sate_mask]

        if len(model_file) == 0:
            print("No corresponding model output.")
            continue
        else:
            model_file = model_file[0]
        # print(f"Processing the model data {orbit}")

        try:
            sate.read_model(model_file)
            model_output, ndacc_model = compare(time_target, sate, ndacc, model=True)
        except Exception as e:
            print(f"Error reading model file: {e}")
            continue

        product_output_list.extend(product_output)
        ndacc_product_list.extend(ndacc_product)

        model_output_list.extend(model_output)
        ndacc_model_list.extend(ndacc_model)

    product_output_mean = np.nanmean(product_output_list)
    ndacc_product_mean = np.nanmean(ndacc_product_list)
    model_output_mean = np.nanmean(model_output_list)
    ndacc_model_mean = np.nanmean(ndacc_model_list)

    product_output_std = np.nanstd(product_output_list)
    ndacc_product_std = np.nanstd(ndacc_product_list)
    model_output_std = np.nanstd(model_output_list)

    return product_output_mean, ndacc_product_mean, model_output_mean, ndacc_model_mean, product_output_std, ndacc_product_std, model_output_std


def compare(time_target, sate, ndacc, model=False):
    # First around the site, the location sift
    lat_dist = np.abs(sate.lat - ndacc.lat)
    lon_dist = np.abs(sate.lon - ndacc.lon)
    dist = 2
    time_dist = 3600 * 1
    if model:
        index_sate = (lat_dist < dist) & (lon_dist < dist) & ((sate.cld == 1) | (sate.cld == 2)) & (sate.uncertainty > 0) & (sate.uncertainty < 4E16)
    else:
        index_sate = (lat_dist < dist) & (lon_dist < dist) & ((sate.cld == 1) | (sate.cld == 2)) & (sate.quality == 2)

    # Then the time overlap
    if np.sum(index_sate) >= 2:         # Define at least 3 points
        time_sate = np.mean(sate.time[index_sate])
        mean_time = sate.mjd2k_to_datetime(time_sate)
        mean_time = mean_time.timestamp()
        index_site = np.abs(ndacc.time - time_sate) < time_dist  # Sift the mean time around observation

        if np.sum(index_site) > 0 :
            era5_data = era5_rttov2()
            era5_data.readcoef(time_target, False)
            surf_temp = era5_data.interpolate(np.array([mean_time]), np.array([ndacc.lat]), np.array([ndacc.lon]), era5_data.skinTemp)[0]
            if surf_temp > 260:
                if model:
                    sate_output = sate.model_co[index_sate]
                else:
                    sate_output = sate.co[index_sate]
                ndacc_output = ndacc.column[index_site]
            else:
                print("The condition on temperature is not satisfied.")
                sate_output = []
                ndacc_output = []
        else:
            sate_output = []
            ndacc_output = []
    else:
        sate_output = []
        ndacc_output = []
    return sate_output, ndacc_output



def process_func(site_name):
    d = begin

    # Read NDACC sites
    ndacc = ndaccSite(site_name)
    ndacc_file = f'/your_ndacc_files/groundbased_ftir.co_{site_name}*'
    ndacc.read_file(ndacc_file)

    time_target_list = []
    product_sate_list = []
    product_ndacc_list = []
    model_sate_list = []

    product_std_list = []
    ndacc_std_list = []
    model_std_list = []

    while d <= end:
        time_target = d.strftime("%Y%m%d")
        time_target_list.append(time_target)
        d += delta

        # Read product data
        product_path = f'/your_product_files/IASI_SND_02_M01_{time_target}*.nc'
        product_path = glob.glob(product_path)
        # Read satellite data
        model_path = f'/your_satellite_files/CO_IASI_xxx_1C_M01_{time_target}*.nc'

        model_path = glob.glob(model_path)
        product_output_mean, ndacc_product_mean, model_output_mean, ndacc_model_mean, product_output_std, ndacc_product_std, model_output_std = siftPoint(time_target, product_path, model_path, ndacc)
        print(f"\033[0;35mDate: {time_target}, Product Mean: {product_output_mean}, NDACC Mean: {ndacc_product_mean}, Model Mean: {model_output_mean}, NDACC Model Mean: {ndacc_model_mean}\033[0m")

        product_sate_list.append(product_output_mean)
        product_ndacc_list.append(ndacc_product_mean)
        model_sate_list.append(model_output_mean)

        product_std_list.append(product_output_std)
        ndacc_std_list.append(ndacc_product_std)
        model_std_list.append(model_output_std)


    time_target_list = np.array(time_target_list)
    product_sate_list = np.array(product_sate_list)
    product_ndacc_list = np.array(product_ndacc_list)
    model_sate_list = np.array(model_sate_list)

    product_std_list = np.array(product_std_list)
    ndacc_std_list = np.array(ndacc_std_list)
    model_std_list = np.array(model_std_list)


    # 创建DataFrame便于绘图
    index_nan = np.isnan(product_sate_list) | np.isnan(model_sate_list) | np.isnan(product_ndacc_list)
    # ratio = 1 if ndacc.unit == "molec cm-2" else 1E18
    ratio = 1 if np.nanmean(product_ndacc_list) < 100 else 1E18
    df = pd.DataFrame({
        'Time': time_target_list[~index_nan],
        'Satellite': product_sate_list[~index_nan] / 1E18,
        'Model': model_sate_list[~index_nan] / 1E18,
        'NDACC': product_ndacc_list[~index_nan] / ratio, 
        'Satellite_STD': product_std_list[~index_nan] / 1E18,
        'Model_STD': model_std_list[~index_nan] / 1E18,
        'NDACC_STD': ndacc_std_list[~index_nan] / (ratio*1E18)
    })
    print(f"\033[0;35mThe final data length for site {site_name} is {len(df)}\033[0m")

    time_array = df['Time'].values
    satellite_array = df['Satellite'].values
    model_array = df['Model'].values
    ndacc_array = df['NDACC'].values
    satellite_std_array = df['Satellite_STD'].values
    model_std_array = df['Model_STD'].values
    ndacc_std_array = df['NDACC_STD'].values

    # 保存为 .npz 文件
    if status == 1:
        title = f'./ref_data/fig/MLP_{site_name}_std.npz'
    if status == 2:
        title = f'./ref_data/fig/{site_name}_std.npz'
        title = f'./ref_data/fig/{site_name}_p.npz'
    np.savez_compressed(title,
                        time=time_array,
                        satellite=satellite_array,
                        model=model_array,
                        ndacc=ndacc_array,
                        satellite_std=satellite_std_array,
                        model_std=model_std_array,
                        ndacc_std=ndacc_std_array)

if __name__ == '__main__':
    begin = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2022, 12, 31)
    delta = datetime.timedelta(1)

    sites = ['kit001_kiruna']
    sites = ['ulg002_jungfraujoch']
    sites = ['spbu001_st.petersburg']
    sites = ['kit002_izana']
    sites = ['niwa004_arrival.heights']
    sites = ['uow002_wollongong']

    sites = ['ncar001_thule']   #
    sites = ['ncar002_mauna.loa.hi']   #
    sites = ['unam001_altzomoni']      #
    sites = ['ncar003_boulder.co']   #
    sites = ['utoronto002_toronto.tao']   #
    all_sites = ['ncar001_thule', 'ncar002_mauna.loa.hi', 'unam001_altzomoni',
                 'ncar003_boulder.co', 'utoronto002_toronto.tao']

    process_func('unam001_altzomoni')
