import os
import pyrttov
import sys
import datetime
from multiprocessing import Pool
import numpy as np
from Tools import irtt_config
import netCDF4 as nc
from scipy.interpolate import interp1d
import numpy as np

class refData():
    filepath = None
    def __init__(self) -> None:
        pass
    
    def nanReplace(self, arr):
        for row in arr:
            if len(row[np.isnan(row)]) > 0:
                last_valid = row[~np.isnan(row)][-1]
                row[np.isnan(row)] = last_valid
        return arr

    def readData(self, params):
        try:
            print('\033[0;33mOpening refdata file %s\033[0m' %self.filepath)
            ncfile = nc.Dataset(self.filepath, mode='r')
            self.lat = np.asarray(ncfile.variables['lat'][:]).astype(float)
            self.lon = np.asarray(ncfile.variables['lon'][:]).astype(float)

            self.time = np.asarray(ncfile.variables['time'][:]).astype(str)
            self.zenith = np.asarray(ncfile.variables['zenith'][:]).astype(float)
            self.azimuth = np.asarray(ncfile.variables['azimuth'][:]).astype(float)
            self.solar_zenith = np.asarray(ncfile.variables['solar_zenith'][:]).astype(float)
            self.solar_azimuth = np.asarray(ncfile.variables['solar_azimuth'][:]).astype(float)

            self.radiance = np.asarray(ncfile.variables['radiance'][:, :])
            self.pressure_profile = np.asarray(ncfile.variables['pressure_profile'][0, :])
            self.temp_profile = np.asarray(ncfile.variables['temp_profile'][:, :])
            self.temp_profile = self.nanReplace(self.temp_profile)  # Replace NaN values with the last valid value
            self.vapor_profile = np.asarray(ncfile.variables['vapor_profile'][:, :])
            self.vapor_profile = self.nanReplace(self.vapor_profile)  # Replace NaN values with the last valid value

            self.surf_z = np.asarray(ncfile.variables['surf_z'][:])
            self.surf_temp = np.asarray(ncfile.variables['surf_temp'][:])
            self.surf_pressure = np.asarray(ncfile.variables['surf_pressure'][:])

            # ERA5 data
            self.surf_type = np.asarray(ncfile.variables['surf_type'][:])
            self.skin_temp = np.asarray(ncfile.variables['skin_temperature'][:])
            self.s2m_temp = np.asarray(ncfile.variables['2m_temperature'][:])
            self.s10m_u_wind = np.asarray(ncfile.variables['10m_u_wind'][:])
            self.s10m_v_wind = np.asarray(ncfile.variables['10m_v_wind'][:])

            # self.surf_emissivity = np.asarray(ncfile.variables['surf_emissivity'][:])
            # self.h2o = np.asarray(ncfile.variables['h2o_column'][:])
            ncfile.close()

            # Read the limits of the RTTOV gas profile
            file_path = f"{params.output_folder}/rtcoef_metop_2_iasi_7gas.h5"
            if params.gas_name == "N2O":
                status = 5
            elif params.gas_name == "CO":
                status = 6
            elif params.gas_name == "CH4":
                status = 7
            ncfile = nc.Dataset(file_path, "r")
            self.env_prfl_gmax = ncfile.groups["COEF"].variables["ENV_PRFL_GMAX"][status, :]  # 廓线最大值
            self.env_prfl_gmin = ncfile.groups["COEF"].variables["ENV_PRFL_GMIN"][status, :]  # 廓线最小值
            self.ref_prfl_p = ncfile.groups["COEF"].variables["REF_PRFL_P"][:]        # 压力廓线
            ncfile.close()

        except Exception as error:
            print(error)


    def generate_profile(self, p_layers):
        """
        插值符合要求的廓线数据，随机生成一个廓线数据
        """
        random_scale = np.random.rand()        # 0-1

        # 创建插值函数
        f_gmin = interp1d(self.ref_prfl_p, self.env_prfl_gmin, kind='linear', fill_value="extrapolate")
        f_gmax = interp1d(self.ref_prfl_p, self.env_prfl_gmax, kind='linear', fill_value="extrapolate")

        # 对目标压力层进行插值
        gmin_interp = f_gmin(p_layers)
        gmax_interp = f_gmax(p_layers)

        profile = gmin_interp + random_scale * (gmax_interp - gmin_interp)
        return profile


    def simulationSpec(self, iasiRttov, params, index):
        simulate_data = {}
        # First recreate our geometry and atmosphere classes
        print(f"The geometry is: \nsolar zenith: {self.solar_zenith[index]}, solar azimuth: {self.solar_azimuth[index]}, \ntime: {self.time[index]}, zenith: {self.zenith[index]}, azimuth: {self.azimuth[index]}")
        
        # Declare an instance of Profiles
        nlevels = len(self.pressure_profile)
        nprofiles = 1
        myProfiles = pyrttov.Profiles(nprofiles, nlevels)

        def expand2nprofiles(n, nprof):
            '''
            Transform 1D array to a [nprof, nlevels] array
            '''
            return np.tile(n, (nprof, 1))

        # Units for gas profiles
        gas_units = 0  # 0 is ppmv over dry air, 1 is kg/kg, 2 is ppmv over moist air

        myProfiles.GasUnits = gas_units
        myProfiles.P = expand2nprofiles(self.pressure_profile, nprofiles)
        myProfiles.T = expand2nprofiles(self.temp_profile[index], nprofiles)
        myProfiles.Q = expand2nprofiles(self.vapor_profile[index], nprofiles)
        
        # Read from the cams target gas profile 
        # Use the model outcome as the target gas profile
        if params.gas_name == "CO":
            co_ex = self.generate_profile(self.pressure_profile)
            myProfiles.CO = expand2nprofiles(co_ex, nprofiles)
            simulate_data['profile'] = co_ex
        elif params.gas_name == "N2O":
            n2o_ex = self.generate_profile(self.pressure_profile)
            myProfiles.N2O = expand2nprofiles(n2o_ex, nprofiles)
            simulate_data['profile'] = n2o_ex


        # Angles
        angles = np.array([[self.zenith[index], self.azimuth[index], self.solar_zenith[index], self.solar_azimuth[index]]], dtype=np.float64)
        myProfiles.Angles = angles

        # s2m[6][nprofiles]: 2m p, 2m t, 2m q, 10m wind u, v, wind fetch
        vapor_nan = np.isnan(self.vapor_profile[index, :])
        s2m = np.array ([[self.surf_pressure[index], self.s2m_temp[index], self.vapor_profile[index, ~vapor_nan][-1], self.s10m_u_wind[index], self.s10m_v_wind[index], None]],dtype=np.float64)
        myProfiles.S2m = s2m

        # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
        skin = np.array([[self.skin_temp[index], None, None, None, None, None, None, None, None]], dtype=np.float64)
        myProfiles.Skin = skin

        # surftype[2][nprofiles]: surftype, watertype
        surftype = np.array([[self.surf_type[index], 1]], dtype=np.int32)
        myProfiles.SurfType = surftype

        # surfgeom[3][nprofiles]: lat, lon, elev
        surfgeom = np.array([[self.lat[index], self.lon[index], self.surf_z[index]]], dtype=np.float64)
        myProfiles.SurfGeom = surfgeom

        time = self.time[index]
        datetimes = np.array([[int(time[0:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12]), int(time[12:14])]], dtype=np.int32)
        myProfiles.DateTimes = datetimes

        # Remove the setting of IASI to the function loadIASI to avoid repeated loading
        # Associate the profiles with each Rttov instance
        iasiRttov.Profiles = myProfiles
        # Load the emissivity and BRDF atlases
        irAtlas = pyrttov.Atlas()
        irAtlas.AtlasPath = '{}/{}'.format(params.rttov_path, "emis_data")
        irAtlas.loadIrEmisAtlas(datetimes[0][1], ang_corr=True) # Include angular correction, but do not initialise for single-instrument

        # Set up the surface emissivity/reflectance arrays and associate with the Rttov objects
        surfemisrefl = np.zeros((5,nprofiles, len(params.channel_list)), dtype=np.float64)
        iasiRttov.SurfEmisRefl = surfemisrefl

        # Call RTTOV
        surfemisrefl[:, :, :] = -1.
        # Call emissivity and BRDF atlases
        try:
            surfemisrefl[0, :, :] = irAtlas.getEmisBrdf(iasiRttov)
        except pyrttov.RttovError as e:
            # If there was an error the emissivities/BRDFs will not have been modified so it
            # is OK to continue and call RTTOV with calcemis/calcrefl set to TRUE everywhere
            sys.stderr.write("Error calling atlas: {!s}".format(e))
        

        # Write the surface emissivity/reflectance to the output dictionary
        # simulate_data['surf_emis'] = np.mean(surfemisrefl[0, 0, :], axis=0)
        try:
            iasiRttov.runDirect()
        except pyrttov.RttovError as e:
            sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
            # Assign -1 to the radiance and weighting matrix
            simulate_data['simulation'] = -np.ones(wavelen_length)
            return simulate_data

        # Write the output to the output file
        simulate_data['simulation'] = np.array(iasiRttov.Rads[0, :])
        return simulate_data


def loadIASI(params):
        # Create Rttov objects for instruments
        # With the target gas
        iasiRttov_with = pyrttov.Rttov()
        iasiRttov_with.FileCoef = f'{params.rttov_path}/rtcoef_rttov13/rttov13pred101L/rtcoef_metop_2_iasi_7gas.H5'
        iasiRttov_with.Options.AddInterp = True
        iasiRttov_with.Options.VerboseWrapper = False
        try:
            iasiRttov_with.loadInst(params.channel_list)
        except pyrttov.RttovError as e:
            sys.stderr.write("Error loading instrument(s): {!s}".format(e))
            sys.exit(1)
        if params.gas_name == "N2O":
            iasiRttov_with.Options.N2OData = True
        elif params.gas_name == "CO":
            iasiRttov_with.Options.COData = True

        return iasiRttov_with


def initNCfile(filepath, points_num, multi_times, extracted_data):
    # 创建目标 NetCDF 文件
    print("Initializing ", filepath)
    output_file = nc.Dataset(filepath, 'w', format='NETCDF4')

    # 创建一个可扩展的维度，用于存储点的数量
    # output_file.createDimension('point', None)
    output_file.createDimension('dim_point', points_num*multi_times)
    output_file.createDimension('dim_radiance', wavelen_length)
    output_file.createDimension('dim_profile', NUM_PRESSURE_PROFILE)

    # 写入波长和波数
    output_file.createVariable('wavelength', 'f4', ('dim_radiance'))
    wave_var = output_file.variables['wavelength']
    wave_var[:] = wavelen[::-1]
    output_file.createVariable('wavenum', 'f4', ('dim_radiance'))
    wave_var = output_file.variables['wavenum']
    wave_var[:] = wavenum[:]

    # 写入经度和纬度
    output_file.createVariable('lat', 'f4', ('dim_point',))
    lat_var = output_file.variables['lat']
    lat_var[:] = np.repeat(extracted_data.lat, multi_times)
    output_file.createVariable('lon', 'f4', ('dim_point',))
    lon_var = output_file.variables['lon']
    lon_var[:] = np.repeat(extracted_data.lon, multi_times)


    # 写入观测光谱的截取部分
    output_file.createVariable('radiance', 'f4', ('dim_point', 'dim_radiance'))
    radiance_var = output_file.variables['radiance']
    radiance_var[:] = np.repeat(extracted_data.radiance, multi_times, axis=0)

    # 写入模拟数据
    output_file.createVariable('simulation', 'f4', ('dim_point', 'dim_radiance'))       # Simulated radiance
    # output_file.createVariable('emissivity', 'f4', ('dim_point', ))       # Emissivity
    output_file.createVariable('profile', 'f4', ('dim_point', 'dim_profile'))       # assigned profile
    output_file.createVariable('pressure_profile', 'f4', ('dim_profile'))       # assigned pressure
    output_file.close()


def writeNCfile(filepath, data, index):
    print("Writing ", filepath)
    with nc.Dataset(filepath, 'a') as output_file:
        # 获取 simulation 变量并写入数据
        simulation_data = data['simulation']
        simulation_var = output_file.variables['simulation']
        simulation_var[index, :] = simulation_data

        profile_data = data['profile']
        profile_var = output_file.variables['profile']
        profile_var[index, :] = profile_data

        # emissivity_data = data['surf_emis']
        # emissivity_var = output_file.variables['emissivity']
        # emissivity_var[index] = emissivity_data


if __name__ == "__main__":
    import glob

    params = irtt_config('../params.json')
    day = params.begin
    MULTI_TIMES = params.multi_times           # For one observation, we can simulate multiple times to get more data
    delta = datetime.timedelta(days=1)

    # Read from the background data and output the simulation
    OUTPUT_IN = f"{params.output_folder}/{params.gas_name}/dataset_extract/raw/"
    OUTPUT_OUT = f"{params.output_folder}/{params.gas_name}/dataset_simulate/raw/"
    NUM_PRESSURE_PROFILE = 37

    wavenum = np.arange(params.wave_num_s, params.wave_num_e + 0.25, 0.25)
    print(f"The calculated wavenumber range is {wavenum}\n")
    wavelen = 1.0e7 / wavenum[::-1]     # cm-1 to nm
    wavelen_length = np.shape(wavelen)[0]

    # The parallel computation version
    func_params = []
    background_paths = glob.glob(f"{OUTPUT_IN}/split_*.nc")
    for filepath in background_paths:
        func_params.append([filepath])


    def process_day(filepath):
        try:
            # Load the IASI Rttov object
            iasiRttov_with = loadIASI(params)

            # Read from the extraction file
            extracted_data = refData()
            extracted_data.filepath = filepath
            extracted_data.readData(params)

            points_num = np.shape(extracted_data.lat)[0]
            print(f"There are {points_num} points in this file, multiply by {MULTI_TIMES} times.")
            filename = os.path.basename(filepath)
            filepath = f'{OUTPUT_OUT}/{filename}'

            # Initialize the output file
            initNCfile(filepath, points_num, MULTI_TIMES, extracted_data)
            # Write the reference pressure profile
            with nc.Dataset(filepath, 'a') as output_file:
                p_profile_var = output_file.variables['pressure_profile']
                p_profile_var[:] = extracted_data.pressure_profile

            for i in range(points_num):
                for j in range(MULTI_TIMES):
                    print(f"\nSimulating {filepath} now, {j+1}/{MULTI_TIMES} times, {i+1}/{points_num}...")

                    # Simulate twice with and without the target gas
                    print("Simulating with the target gas...")
                    simulate_data_with = extracted_data.simulationSpec(iasiRttov_with, params, i)
                    writeNCfile(filepath, simulate_data_with, i*MULTI_TIMES+j)

        except Exception as error:
            print(error)

    with Pool(processes=40) as pool:
        pool.starmap(process_day, func_params)
