'''
Grid the data from the Level2 data to the Level3 data
'''
import numpy as np
from glob import glob
import datetime
import harp
import h5py

def denormalize(data, min, max, scale=1, log=False):
    data = (data + 1) / 2
    data = data * (max - min) + min 
    if log:
        data = np.exp(data)
    data = data / scale
    return data

def read_ozprof_data(filepath):
    print('Reading ', filepath)
    data_ret = {}
    key_component = {f"{gas}": 'column', "lat": "lat", "lon": "lon", "cld": "cld", "uncertainty": "uncertainty", "surfP": "input_data", "surfT": "input_data"}
    try:
        with h5py.File(filepath, 'r') as f:
            for name, attr in key_component.items():
                if name == 'surfP':
                    data = f[attr][:, :, -1]
                elif name == 'surfT':
                    data = f[attr][:, :, -2]
                else:
                    data = f[attr][:]
                data_ret[name] = data[:-1,:-1]

                if name == 'lat':
                    lat_bounds = np.zeros((data.shape[0] - 1, data.shape[1] - 1, 4))
                    lat_bounds[:, :, 0] = data[:-1, :-1]
                    lat_bounds[:, :, 1] = data[1:, :-1]
                    lat_bounds[:, :, 2] = data[1:, 1:]
                    lat_bounds[:, :, 3] = data[:-1, 1:]
                    data_ret['Latitude_bounds'] = lat_bounds
                elif name == 'lon':
                    lon_bounds = np.zeros((data.shape[0] - 1, data.shape[1] - 1, 4))
                    lon_bounds[:, :, 0] = data[:-1, :-1]
                    lon_bounds[:, :, 1] =  data[1:, :-1]
                    lon_bounds[:, :, 2] =  data[1:, 1:]
                    lon_bounds[:, :, 3] =  data[:-1, 1:]
                    data_ret['Longitude_bounds'] = lon_bounds

    except Exception as error:
        print('\033[0;31mOpening file error! %s\033[0m' %error)
        return None
    return data_ret


def convert_ozprof_data(data):

    index=np.asarray(np.where ( (data['lat']>=-90) & (data['lat']<=90) & (data['lon']>=-180) & (data['lon']<=180) \
                                & (data['Latitude_bounds'][:,:,1]>-90) & (abs(data['Longitude_bounds'][:,:,2]-data['Longitude_bounds'][:,:,0])<1) \
                                & (abs(data['Latitude_bounds'][:, :, 2] - data['Latitude_bounds'][:, :, 0]) < 1)))
    xy = np.shape(index)[1]
    print(np.shape(index))

    product = harp.Product()

    product['cld'] = harp.Variable(data=data['cld'][index[0,:],index[1,:]].flatten(), dimension=['time'], unit="",
                                valid_min=1, valid_max=4, description="IASI Cloud Flag")
    product[f'{gas}'] = harp.Variable(data=data[f'{gas}'][index[0,:],index[1,:]].flatten(), dimension=['time'], unit="",
                                valid_min=-2E18, valid_max=10E18, description=f"Concentration of {gas}")
    product['uncertainty'] = harp.Variable(data=data['uncertainty'][index[0,:],index[1,:]].flatten(), dimension=['time'], unit="",
                                valid_min=-1E19, valid_max=1E19, description=f"Uncertainty of {gas}")
    product['surfT'] = harp.Variable(data=data['surfT'][index[0,:],index[1,:]].flatten(), dimension=['time'], unit="",
                                valid_min=210, valid_max=360, description="Surface Temperature")
    product['latitude_bounds'] = harp.Variable(data=data['Latitude_bounds'][index[0,:],index[1,:],:].reshape((xy, 4)), dimension=['time', None], unit="degrees_north",
                                valid_min=-90.0, valid_max=90.0, description="Geodetic Corner Latitude Bounds")
    product['longitude_bounds'] = harp.Variable(data=data['Longitude_bounds'][index[0,:],index[1,:],:].reshape((xy, 4)), dimension=['time', None], unit="degrees_east",
                                valid_min=-180.0, valid_max=180.0, description="Geodetic Corner Longitude Bounds")
    return product


def merge_ozprof_data(data1, data2):
    if data1 and data2:
        data = {}
        for k in data1.keys():
            data[k] = np.append(data1[k], data2[k], axis=0)
    elif data1 and not data2:
        data = data1
    elif data2 and not data1:
        data = data2
    else:
        data = None
    return data


def regrid_ozprof(product, output, resolution, region):
    lat_0 = region[0]
    lon_0 = region[2]
    lat_length = int((region[1] - region[0])/resolution) + 1
    lon_length = int((region[3] - region[2])/resolution) + 1

    print('\033[0;32mWriting file to %s\n\033[0m' %output)
    operations = ";".join([
        f"cld>=1;cld<={cf};uncertainty<=4E16;uncertainty>=0;CO<=10E18;CO>=-2E18;surfT>260",
        f"keep(latitude_bounds,longitude_bounds,{gas},cld, uncertainty, surfT)",
        "bin_spatial({},{},{},{},{},{})".format(lat_length, lat_0, resolution, lon_length, lon_0, resolution),
        "derive(latitude {latitude})",
        "derive(longitude {longitude})",
    ])
    regridded_product = harp.execute_operations(product, operations)
    harp.export_product(regridded_product, output)
    return regridded_product

def grid_tropomi_ozprof(l2files, output, resolution, region):
    products = []
    j = 0
    for i, f in enumerate(l2files):
        try:
            if j == 0:
                data = read_ozprof_data(f)
            else:
                data = merge_ozprof_data(data, read_ozprof_data(f))
            j+=1

        except Exception as error:
            print('\033[0;31mReading error! %s\033[0m' %error)

    try:
        product = convert_ozprof_data(data)
        products.append(product)
        product = harp.concatenate(products)

    except Exception as error:
        print('\033[0;31mProcessing error! %s\033[0m' %error)

    regrid_ozprof(product, output, resolution, region)

if __name__ == '__main__':
    begin = datetime.date(2022, 1, 1)
    end = datetime.date(2022, 1, 1)
    d = begin
    delta = datetime.timedelta(days=1)

    resol = 0.2
    cf = 2
    appendix = '_grid02cf2.nc'
    gas = 'CO'

    while d <= end:
        time_target = d.strftime("%Y%m%d")
        d += delta
        print(time_target)
        pathL2 = f'/your_l2_path/{gas}_IASI_xxx_1C_M01_' + time_target + '*.nc'
        pathOut = f'/your_l3_path/MetopB_{gas}_' + time_target + appendix

        files = sorted(glob(pathL2))
        if len(files) == 0:
            continue
        try:
            grid_tropomi_ozprof(l2files=files, output=pathOut, resolution=resol, region=[-90, 90, -180, 180])
        except Exception as error:
            print('\033[0;31mError! %s\033[0m' %error)
