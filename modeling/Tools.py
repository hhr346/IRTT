import datetime
import netCDF4
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class irtt_config:
    def __init__(self, path):
        params = self.load_config(path)
        # Time
        self.begin = datetime.datetime.strptime(params['date_begin'], "%Y-%m-%d").date()
        self.end = datetime.datetime.strptime(params['date_end'], "%Y-%m-%d").date()
        
        # Names
        self.gas_name = params[params["gas_id"]]["name"]
        self.sate_name = params["satellite"]
        self.multi_times = int(params["multiple_time"])

        # Read channels
        self.wave_num_s, self.wave_num_e = params[params["gas_id"]]["range"]
        wavenumber = np.arange(645.0, 2760.0, 0.25)
        start_idx = np.where(wavenumber == self.wave_num_s)[0][0]
        end_idx = np.where(wavenumber == self.wave_num_e)[0][0] + 1
        self.wavenumber = wavenumber[start_idx:end_idx]

        window = np.array([self.wave_num_s, self.wave_num_e])
        chan_window = (window-645)*4 + 1 
        self.channel_list = range(chan_window[0], chan_window[1]+1)

        # Paths
        self.output_folder = params["ref_out"]
        self.rttov_path = params["rttov_path"]
        self.level1_path = params["L1_data"]
        self.level2_path = params["L2_data"]

    def load_config(self, file_path):
        import json
        # 打开并读取 JSON 配置文件
        with open(file_path, 'r') as file:
            config = json.load(file)  # 使用 json.load() 方法解析文件内容
        return config


class emis_uw:
    def __init__(self):
        self.lat = np.arange(89.975, -90, -0.05)
        self.lon = np.arange(-179.975, 180, 0.05)

    def interpolate(self, new_wavenum, new_lat, new_lon, data):
        from scipy.interpolate import RegularGridInterpolator
        # 创建插值函数
        interpolator = RegularGridInterpolator((self.wavenumber, self.lon, self.lat), data, bounds_error=False, fill_value=None)
        dim = new_lat.ndim
        if dim == 1 or dim == 0:
            query_points = np.array(list(zip(new_wavenum, new_lat, new_lon)))
            new_data = interpolator(query_points)
        elif dim == 2:
            query_points = np.array(list(zip(new_wavenum.flatten(), new_lat.flatten(), new_lon.flatten())))
            new_data_flat = interpolator(query_points)
            new_data = new_data_flat.reshape(new_lat.shape[0], new_lat.shape[1])
        return new_data

    def readcoef(self, date):
        month = date[4:6]
        filepath = f"/exports/d3/hhr346/uw_emis/global_emis_2016{month}.nc"
        try:
            print('Opening emis file ', filepath)
            ncfile = netCDF4.Dataset(filepath)
            self.wavenumber = np.asarray(ncfile.variables['wavenumber'][:4])
            emis_flag = np.asarray(ncfile.variables['emis_flag'][:, :])
            emis1 = np.asarray(ncfile.variables['emis1'][:, :])
            emis2 = np.asarray(ncfile.variables['emis2'][:, :])
            emis3 = np.asarray(ncfile.variables['emis3'][:, :])
            emis4 = np.asarray(ncfile.variables['emis4'][:, :])

            self.emis = np.stack((emis1, emis2, emis3, emis4), axis=0)
            self.emis[:, emis_flag == 0] = 0.99
            ncfile.close()
        except Exception as err:
            print(err)


class era5_rttov:
    def __init__(self):
        self.lat = np.arange(-90, 90.25, 0.25)
        self.lon = np.arange(-179.75, 180.25, 0.25)

    def interpolate(self, new_time, new_lat, new_lon, data):
        from scipy.interpolate import RegularGridInterpolator
        # 创建插值函数
        interpolator = RegularGridInterpolator((self.time, self.lat, self.lon), data, bounds_error=False, fill_value=None)
        dim = new_lat.ndim
        if dim == 1 or dim == 0:
            query_points = np.array(list(zip(new_time, new_lat, new_lon)))
            new_data = interpolator(query_points)

        elif dim == 2:
            query_points = np.array(list(zip(new_time.flatten(), new_lat.flatten(), new_lon.flatten())))
            new_data_flat = interpolator(query_points)
            if data.ndim == 3:
                new_data = new_data_flat.reshape(new_lat.shape[0], new_lat.shape[1])
            elif data.ndim == 4:
                new_data = new_data_flat.reshape(new_lat.shape[0], new_lat.shape[1], data.shape[-1])
        return new_data

    def convert(self, data):
        # Convert the data from the original format to the twisted format
        if data.ndim == 3:
            data1 = data[:, :, :721]
            data2 = data[:, :, 721:]
        elif data.ndim == 2:
            data1 = data[:, :721]
            data2 = data[:, 721:]
        data = np.concatenate((data2, data1), axis=-1)
        data = np.flip(data, axis=-2)
        return data

    def convert_profile(self, data):
        # Convert the data from the original format to the twisted format
        data1 = data[:, :, :, :721] 
        data2 = data[:, :, :, 721:]
        data = np.concatenate((data2, data1), axis=-1)   # Concatenate the longitude
        data = np.flip(data, axis=-2)   # Flip the latitude
        data = np.flip(data, axis=1)   # Flip the pressure
        data = data.transpose(0, 2, 3, 1) # Rearrange the dimension, (time, 721, 1440, 37)
        return data

    def convertVapor(self, q):
        """
        Convert specific humidity (q) to Volumn mixing ratio (r).
        Parameters:
            q (float or ndarray): Specific humidity (kg/kg).
        Returns:
            r (float or ndarray): Mixing ratio (ppmv).
        """
        q = q / (1 - q)

        M_dry_air = 28.97  # 干空气的摩尔质量 (g/mol)
        M_water_vapor = 18.02  # 水汽的摩尔质量 (g/mol)
        conversion_factor = (M_dry_air / M_water_vapor) * 1e6
        ppmv = q * conversion_factor
        return ppmv

    def readcoef(self, date, profile=True):
        filepath = f"/exports/d3/hhr346/era5/{date[:4]}/daily_{date}.nc"
        print('Opening era5 file ', filepath)
        ncfile = netCDF4.Dataset(filepath)
        self.time = np.asarray(ncfile.variables['valid_time'][:])

        surfaceType = ncfile.variables['lsm'][:]
        surfaceType = self.convert(surfaceType)    # 0 means ocean, >1 means land
        self.surfaceType = np.where(surfaceType > 0, 0, 1)    # 1 means ocean, 0 means land

        skinTemp = ncfile.variables['skt'][:]
        self.skinTemp = self.convert(skinTemp)

        s2mTemp = ncfile.variables['t2m'][:]
        self.s2mTemp = self.convert(s2mTemp)

        s10mVwind = ncfile.variables['v10'][:]
        self.s10mVwind = self.convert(s10mVwind)

        s10mUwind = ncfile.variables['u10'][:]
        self.s10mUwind = self.convert(s10mUwind)

        surfaceZ = ncfile.variables['z'][:]
        surfaceZ = surfaceZ / (9.81 * 1e3)      # Geopotential Height to Geometric Height, m2/s2 -> km
        self.surfaceZ = self.convert(surfaceZ)

        surfPressure = ncfile.variables['sp'][:]
        surfPressure = self.convert(surfPressure)
        self.surfPressure = surfPressure / 100.0    # Pa to hPa
        ncfile.close()

        if profile:
            filepath = f"/exports/d3/hhr346/era5/2022/profile_{date}.nc"
            try:
                print('Opening profile file ', filepath)
                ncfile = netCDF4.Dataset(filepath)

                # Pressure Profile
                self.pressure_p = np.asarray(ncfile.variables['pressure_level'][::-1])    # hPa, (37), from top to bottom, increasing
                # Temperature Profile
                self.temp_p = np.asarray(ncfile.variables['t'][:, :, :, :])    # K, (time, 37, 721, 1440)
                self.temp_p = self.convert_profile(self.temp_p)
                # Water Vapor Profile, convert from Specifc Humidity to Mixing Ratio
                self.vapor_p = np.asarray(ncfile.variables['q'][:, :, :, :])    # Specific Humidity, kg/kg, (365, 37, 721, 1440)
                self.vapor_p = self.convert_profile(self.vapor_p)
                self.vapor_p = self.convertVapor(self.vapor_p)   # Unit Convertion, Mixing Ratio, ppmv
            except Exception as err:
                print(err)


class iasi:
    time_target = None
    path1level = None
    path2level = None
    path3level = None
    name = 'IASI'

    def __init__(self, name) -> None:
        iasi.name = name
        self.lat = None
        self.lon = None
        self.scale_index = np.array([2581, 5921, 9009, 9541, 10721])
        self.scale_index = self.scale_index - 2581
        self.scale_index = np.array([int(i) for i in self.scale_index])

        self.scale_factor = np.array([7, 8, 9, 8, 9])
        self.scale_factor = self.scale_factor-3-2         # W/(m^2 sr m^-1) to mW/(m^2 sr cm^-1)

    def convert_to_datetime(self, num):
        base_date = datetime.datetime(2000, 1, 1)
        # Convert to datetime
        date = np.vectorize(lambda x: base_date + datetime.timedelta(seconds=int(x)))(num)
        # Convert to string
        date = np.vectorize(lambda x: x.strftime('%Y%m%d%H%M%S'))(date)
        return date

    def convert_to_unixtime(self, num):
        '''
        From the base of 2000-01-01 00:00:00 to the unix time (1970-01-01 00:00:00)
        '''
        delta_seconds = (datetime.datetime(2000, 1, 1) - datetime.datetime(1970, 1, 1)).total_seconds()
        unix_time = np.vectorize(lambda x: int(x + delta_seconds))(num)
        return unix_time

    def datetime_to_mjd(self, dt):
        # MJD 0 对应的日期是 1858 年 11 月 17 日
        base_mjd = datetime.datetime(1858, 11, 17)
        mjd = (dt - base_mjd).days
        return mjd
    
    def convert_to_mjd(self):
        return np.vectorize(lambda x: self.datetime_to_mjd(self.convert_to_datetime(x)))

    def convert_kgkg_to_ppmv(self, q):
        """
        将水汽的质量混合比 (kg/kg) 转换为体积混合比 (ppmv)。
        
        参数:
            q (float or np.ndarray): 质量混合比 (kg/kg)。
            
        返回:
            np.ndarray: 体积混合比 (ppmv)。
        """
        M_dry_air = 28.97  # 干空气的摩尔质量 (g/mol)
        M_water_vapor = 18.02  # 水汽的摩尔质量 (g/mol)
        conversion_factor = (M_dry_air / M_water_vapor) * 1e6
        
        ppmv = q * conversion_factor
        return ppmv

    def compute_column_density_from_ppm(self, p, ppm, T):
        """
        将 ppm 转为分子柱密度 (molec/cm²)。
        input:
        p   : pressure profile (Pa), 从顶部到地表递增
        ppm : volume mixing ratio (ppm)
        T   : temperature profile (K)
        """
        # 常量
        R = 8.314  # 通用气体常数 (J/(mol·K))
        N_A = 6.02214e23  # 阿伏伽德罗常数 (molec/mol)
        # 气压差 Δp
        delta_p = p[1:] - p[:-1]  # 长度 N-1
        # ppm 转为体积分数 (fractional volume mixing ratio)
        ppm_fraction = ppm / 1e6
        # 每层的气体柱密度 ΔN (molec/cm²)
        delta_N_cm = (delta_p / (R * T[:-1])) * ppm_fraction[:-1] * N_A / 1e4  # 转为 molec/cm²

        # 总列浓度
        total_column_density = np.sum(delta_N_cm)
        return total_column_density

    def compute_column_density_from_profile(self, p, q, M_gas):
        """
        将质量混合比转为分子柱密度 (molec/cm²)。
        input:
        P   : pressure profile (Pa)
        Q   : mass mixing ratio (kg/kg)
        M_gas: The molar mass of the gas (g/mol)
        """
        # 常量
        g = 9.80665  # 重力加速度 (m/s^2)
        N_A = 6.02214e23  # 阿伏伽德罗常数 (molec/mol)
        # 气压差 Δp
        delta_p = p[1:] - p[:-1]
        # 混合比平均值
        q_avg = (q[:-1] + q[1:]) / 2
        # 每层的分子柱密度 ΔN (molec/cm²)
        delta_N_cm = (q_avg * delta_p / (g * M_gas)) * N_A / 1e4  # 转换为 molec/cm²

        # 整层总列密度
        total_column_density = np.sum(delta_N_cm)
        return total_column_density

    def kg_per_m2_to_molecules_per_m2(mass_kg_per_m2, M_gas):
        """
        将 kg/m² 转换为molec/m²
        
        参数:
            mass_kg_per_m2 (float): CO 的质量密度，单位 kg/m²
            
        返回:
            float: 分子数量/m²
        """
        avogadro_number = 6.022e23          # 阿伏伽德罗常数 (分子/mol)
        molecules_per_m2 = (mass_kg_per_m2 / M_gas) * avogadro_number
        return molecules_per_m2

    def read1Level(self, read_radiance=True):
        try:
            print('\033[0;33mOpening l1 file %s\033[0m' %self.path1level)
            ncfile = netCDF4.Dataset(self.path1level, mode='r')
            self.lat = np.asarray(ncfile.variables['lat'][:, :, :]).reshape(-1, 120)
            self.lon = np.asarray(ncfile.variables['lon'][:, :, :]).reshape(-1, 120)

            if read_radiance:
                radiance = np.asarray(ncfile.variables['gs_1c_spect'][:, :, :, :8461])
                radiance = radiance.reshape(-1, 120, 8461)
                num_cols = radiance.shape[-1]

                self.radiance = np.zeros_like(radiance, dtype=np.float32)
                # 遍历分段进行缩放
                for i, start_idx in enumerate(self.scale_index):
                    end_idx = self.scale_index[i + 1] if i + 1 < len(self.scale_index) else num_cols  # 下一段的开始或末尾
                    scale = 10. ** (-self.scale_factor[i])
                    self.radiance[:, :, start_idx:end_idx] = radiance[:, :, start_idx:end_idx] * scale

            self.wavenumber = np.linspace(645, 645+0.25*8460, 8461)

            # 对时间维度单独处理 (xxx, 30) -> (xxx, 120)
            self.time = np.asarray(ncfile.variables['measurement_date'][:, :])
            self.unixtime = self.convert_to_unixtime(self.time)
            self.unixtime = np.repeat(self.unixtime, 4, axis=1)

            self.time = self.convert_to_datetime(self.time)
            # self.time = self.convert_to_mjd()(self.time)
            self.time = np.repeat(self.time, 4, axis=1)

            # Angles
            self.azimuth = np.asarray(ncfile.variables['pixel_azimuth_angle'][:, :, :]).reshape(-1, 120)
            self.zenith = np.asarray(ncfile.variables['pixel_zenith_angle'][:, :, :]).reshape(-1, 120)
            self.solar_azimuth = np.asarray(ncfile.variables['pixel_solar_azimuth_angle'][:, :, :]).reshape(-1, 120)
            self.solar_zenith = np.asarray(ncfile.variables['pixel_solar_zenith_angle'][:, :, :]).reshape(-1, 120)
            ncfile.close()

        except Exception as err:
            print(f"Opening Level1 data error: {err}")
            return 1
        return 0

    def read2Level(self):
        try:
            print('\033[0;33mOpening l2 file %s\033[0m' %self.path2level)
            ncfile = netCDF4.Dataset(self.path2level, mode='r')
            
            # Profile
            self.pressure_profile = np.asarray(ncfile.variables['pressure_levels_temp'][:97], dtype=np.float32) # Pa, (101)
            self.pressure_profile = self.pressure_profile / 100. # Pa to hPa

            self.temperature_profile = np.asarray(ncfile.variables['atmospheric_temperature'][:, :, :97], dtype=np.float32) # K, (765, 120, 101)
            self.temperature_profile[(self.temperature_profile < 0) | (self.temperature_profile > 400)] = np.nan

            self.vapor_profile = np.asarray(ncfile.variables['atmospheric_water_vapor'][:, :, :97], dtype=np.float32) # kg/kg, (765, 120, 101)
            self.vapor_profile[(self.vapor_profile < 0) | (self.vapor_profile > 2)] = np.nan
            self.vapor_profile = self.convert_kgkg_to_ppmv(self.vapor_profile)      # kg/kg to ppmv

            # Surface
            # self.surface_temp = np.asarray(ncfile.variables['surface_temperature'][:, :], dtype=np.float32)   # K, (765, 120)
            # self.surface_temp[(self.surface_temp < 200) | (self.surface_temp > 400)] = np.nan
            # self.surface_z = np.asarray(ncfile.variables['surface_z'][:, :])    # m, (765, 120)
            # self.surface_z = self.surface_z/1000 # m to km

            # self.surface_pressure = np.asarray(ncfile.variables['surface_pressure'][:, :], dtype=np.float32)      # Pa, (765, 120)
            # self.surface_pressure[(self.surface_pressure < 4E4) | (self.surface_pressure > 12E4)] = np.nan
            # self.surface_pressure = self.surface_pressure / 100. # Pa to hPa

            # self.surface_emissivity = np.asarray(ncfile.variables['surface_emissivity'][:, :, 0])
            
            # Gases
            # self.h2o = np.asarray(ncfile.variables['integrated_water_vapor'][:, :])
            # self.n2o = np.asarray(ncfile.variables['integrated_n2o'][:, :])
            # self.co = np.asarray(ncfile.variables['integrated_co'][:, :])
            # self.co2 = np.asarray(ncfile.variables['integrated_co2'][:, :])
            # self.ch4 = np.asarray(ncfile.variables['integrated_ch4'][:, :])

            self.cld = np.asarray(ncfile.variables['flag_cldnes'][:, :])   # 1-4, 1 and 2 can be seen as clear
            # Geolocation
            self.lat = np.asarray(ncfile.variables['lat'][:, :])
            self.lon = np.asarray(ncfile.variables['lon'][:, :])

            # 数据筛选
            ncfile.close()

        except Exception as err:
            print(f"Opening Level2 data error: {err}")

    def read3Level(self):
        try:
            print('Opening l3 file ', self.path3level)
            ncfile = netCDF4.Dataset(self.path3level, mode='r')

            self.lon = np.asarray(ncfile.variables['lon'][:]) 
            self.lat = np.asarray(ncfile.variables['lat'][:])
            self.hri = np.asarray(ncfile.variables['HRI'][:, :])
            ncfile.close()

        except Exception as error:
            print('\033[0;31mReading LEVEL3 data error: %s\033[0m' %error)
        

class plotTools:
    # Map area, lon first
    globe = [-180, 180, -90, 90]
    china = [70, 135, 15, 55]
    usa = [-130, -60, 20, 60]
    africa = [-30, 62, -40, 40]

    shandong = [110, 123, 30, 42]
    yrd = [117, 123, 28, 34]            # Yangze River Delta
    mada = [40, 55, -30, -10]
    moz = [30, 42, -25, -10]

    south_africa = [30, 55, -30, -10]
    south_asia = [65, 135, 5, 55]
    south_america = [-80, -35, -60, 15]

    gems = [70, 140, -10, 50]
    pacific = [-180, -90, -90, 90]

    tccon_sites = {}
    # Asia
    tccon_sites['hf'] = {"name": "Hefei", "Continent": "Asia"}
    tccon_sites['xh'] = {"name": "Xianghe", "Continent": "Asia"}
    tccon_sites['rj'] = {"name": "Rikubetsu", "Continent": "Asia"}
    tccon_sites['tk'] = {"name": "Tsukuba", "Continent": "Asia"}
    tccon_sites['js'] = {"name": "Saga", "Continent": "Asia"}
    tccon_sites['bu'] = {"name": "Burgos", "Continent": "Asia"}

    # Australia
    tccon_sites['db'] = {"name": "Darwin", "Continent": "Australia"}
    tccon_sites['wg'] = {"name": "Wollongong", "Continent": "Australia"}
    tccon_sites['lh'] = {"name": "Lauder", "Continent": "Australia"}

    # Africa, South America
    tccon_sites['ra'] = {"name": "Reunion", "Continent": "Africa"}
    tccon_sites['iz'] = {"name": "Izana", "Continent": "Africa"}
    tccon_sites['ni'] = {"name": "Nicosia", "Continent": "Europe"}
    tccon_sites['ma'] = {"name": "Manaus", "Continent": "SouthAmerica"}

    # North America
    tccon_sites['pa'] = {"name": "Park Falls", "Continent": "NorthAmerica"}
    tccon_sites['if'] = {"name": "Indianapolis", "Continent": "NorthAmerica"}
    tccon_sites['oc'] = {"name": "Lamont", "Continent": "NorthAmerica"}
    tccon_sites['et'] = {"name": "East Trout Lake", "Continent": "NorthAmerica"}
    tccon_sites['fc'] = {"name": "Four Corners", "Continent": "NorthAmerica"}
    tccon_sites['eu'] = {"name": "Eureka", "Continent": "NorthAmerica"}
    tccon_sites['df'] = {"name": "Edwards", "Continent": "NorthAmerica"}
    tccon_sites['ci'] = {"name": "Pasadena", "Continent": "NorthAmerica"}
    tccon_sites['jc'] = {"name": "JPL", "Continent": "NorthAmerica"}
    tccon_sites['jf'] = {"name": "JPL", "Continent": "NorthAmerica"}
    
    # Europe
    tccon_sites['ny'] = {"name": "Ny-Alesund", "Continent": "Europe"}
    tccon_sites['so'] = {"name": "Sodankyla", "Continent": "Europe"}
    tccon_sites['br'] = {"name": "Bremen", "Continent": "Europe"}
    tccon_sites['hw'] = {"name": "Harwell", "Continent": "Europe"}
    tccon_sites['pr'] = {"name": "Paris", "Continent": "Europe"}
    tccon_sites['or'] = {"name": "Orleans", "Continent": "Europe"}
    tccon_sites['gm'] = {"name": "Garmisch", "Continent": "Europe"}
    tccon_sites['ka'] = {"name": "Karlsruhe", "Continent": "Europe"}


    @classmethod
    def getColormap(cls, type='own'):
        if type == 'own':
            cmap_value = np.loadtxt('../common/colormap_22.txt')
            cmap = []
            for i in range(cmap_value.shape[0]):
                cmap.append((cmap_value[i, 0], cmap_value[i, 1], cmap_value[i, 2]))
            return matplotlib.colors.LinearSegmentedColormap.from_list('test_cmap', cmap, N=cmap_value.shape[0])
        else:
            return matplotlib.cm.get_cmap(name=type)
    
