# README

Instructions for the codes and materials of IRTT paper *Transformer-based Inverse Radiance Transfer Framework for Global Carbon Monoxide Retrieval Using Metop-B/IASI*.

The dataset, monthly and yearly distribution of this study are openly available in Zenodo at https://doi.org/10.5281/zenodo.20647547 （without surface temperature） and https://doi.org/10.5281/zenodo.21961015 (with surface temperature）



## Download Scripts

### IASI

Refer to the instructions from the official sites: 

[index.ipynb · master · EUMETlab / Data Services / eumdac_data_store · GitLab (eumetsat.int)](https://gitlab.eumetsat.int/eumetlab/data-services/eumdac_data_store/-/blob/master/index.ipynb)

[Data Tailor Standalone guide | EUMETSAT - User Portal](https://user.eumetsat.int/resources/user-guides/data-tailor-standalone-guide) 

[Data Store detailed guide | EUMETSAT - User Portal](https://user.eumetsat.int/resources/user-guides/data-store-detailed-guide#DataStoredetailedguide-Outputformats)

[EUMETSAT Data Access Client (EUMDAC) guide | EUMETSAT - User Portal](https://user.eumetsat.int/resources/user-guides/eumetsat-data-access-client-eumdac-guide#ID-Data-Tailor-Standalone-in-EUMDAC)



The corresponding script is `download_iasi.py`, first complete the user-specific secret key and secret, then the requests can be customized in the following parameters. 

```python
consumer_key = 'xxx'
consumer_secret = 'xxx'

# Set sensing start and end time
'''
STATUS for the product name
0: L1C
1: PCS # Not working
2: L2
change here
'''
status = 1
start = datetime.datetime(2022, 1, 1)
end = datetime.datetime(2022, 1, 2)
satellite_type = "Metop-B"
```

The api collections are listed as:

[api.eumetsat.int/data/browse/collections?format=html](https://api.eumetsat.int/data/browse/collections?format=html)



### ERA5

Refer to [Catalogue — Climate Data Store](https://cds.climate.copernicus.eu/datasets) 

The download script for the profiles is `download_era5.py`, first install the `cdsapi` and complete the secret code file `~/.cdsapirc`, then the requests can be customized in the following parameters. 

```
url: https://cds.climate.copernicus.eu/api
key: xxxxxxxxxx
```

```python
        begin = datetime.date(2022, 1, 1)
        end = datetime.date(2022, 1, 1)
        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "specific_humidity",
                "temperature"
            ],
            "year": [year],
            "month": [month],
            "day": [day],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "pressure_level": [
                "1", "2", "3",
                "5", "7", "10",
                "20", "30", "50",
                "70", "100", "125",
                "150", "175", "200",
                "225", "250", "300",
                "350", "400", "450",
                "500", "550", "600",
                "650", "700", "750",
                "775", "800", "825",
                "850", "875", "900",
                "925", "950", "975",
                "1000"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }
```



## IRTT pipeline

The whole pipeline is realized in the `modeling` file folder. First switch into the `modeling` by `cd modeling`, and then `python xx.py` after set up the proper environment. 

`1_extractDataset.py` extracts some points from the orbits randomly. 

`2_sum_sift_split.py` summarizes all the points and then split the observation points by the surface temperature.

`3_simulateOnce.py` calls the RTTOV python wrapper to simulate spectrum under various circumstances. 

> Install and set up RTTOV first, details in [RTTOV Downloads | NWP SAF](https://nwp-saf.eumetsat.int/site/software/rttov/download/) 

`4_calLabel.py` calculates the columns from profiles. 

`5_sumDataset.py` generates the final dataset. 



Finally, the training code is shown in `train_p.py` and  `train_temp_noise.py` (with or without surface temperature), they will train the trained models with different parameters to tune and generate `*.pth` and `*.npz` files in `model_path` file folder.

`6_extractOrbit.py` extracts inputs for the trained model from all the orbits that need to be retrieved. 

The evaluation code is shown in the `eval.py`, and it uses the trained model weights to do inference on the orbits. You can change the used model by switching the variable `suffix` and the `import ...` should also be changed respectively. 

```python
from train_p import EncoderRegressor, apply_normalizer
suffix = '_final_p_300'

from train_temp_noise import EncoderRegressor, apply_normalizer
suffix = '_final_t_300'
```



## Gridding Codes

The Level2 outputs are gridded into Level3 files with `harp` library, first install the `harp` library and use the code of `grid_iasi_model.py`. The sifting codes are as below:

```python
operations = ";".join([
        f"cld>=1;cld<={cf};uncertainty<=4E16;uncertainty>=0;CO<=10E18;CO>=-2E18;surfT>260",
        f"keep(latitude_bounds,longitude_bounds,{gas},cld, uncertainty, surfT)",
        "bin_spatial({},{},{},{},{},{})".format(lat_length, lat_0, resolution, lon_length, lon_0, resolution),
        "derive(latitude {latitude})",
        "derive(longitude {longitude})",
    ])
```

You can change the resolution with `resol` and sifting conditions in the gridding code. The code will process the  files in `pathL2` and output in `pathOut`. 

The gridding code file for the official products is `grid_iasi_product.py`. Most of the codes are the same with `grid_iasi_model.py`.



## Comparing Codes

The comparison between the model outputs or product files and the ground-based observations (TCCON and NDACC) are `compare_tccon_npz.py` and `compare_ndacc_npz.py`. The codes will output an `npz` file for each site with key information.


