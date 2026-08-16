"""
然后根据合并的数据集，绘制分布图来观察输入数据的分布情况
按照区间将数据下采样，获得均匀分布的数据集
"""
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from Tools import irtt_config
import datetime
import glob

def split_netcdf(input_file, output_dir, num_splits):
    """
    将NetCDF文件拆分成指定数量的小文件
    
    参数:
        input_file (str): 输入NetCDF文件路径
        output_dir (str): 输出目录
        num_splits (int): 要拆分的文件数量
    """
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_dataset(input_file)
    
    # 获取时间维度
    time_dim = 'dim_point' if 'dim_point' in ds.dims else next(iter(ds.dims))
    time_values = ds[time_dim]
    total_length = len(time_values)
    
    # 计算每个分块的大小
    chunk_size = total_length // num_splits
    remainder = total_length % num_splits
    
    start = 0
    for i in range(num_splits):
        # 计算当前分块的结束位置
        end = start + chunk_size + (1 if i < remainder else 0)
        # 选择当前分块的数据
        chunk = ds.isel({time_dim: slice(start, end)})
        # 生成输出文件名（3位数字，左边补零）
        output_file = os.path.join(output_dir, f"split_{i:03d}.nc")
        # 保存分块到新文件
        chunk.to_netcdf(output_file)
        print(f"Saved {output_file}")
        # 更新下一个分块的起始位置
        start = end


if __name__ == "__main__":
    # 读取配置文件
    params = irtt_config('../params.json')
    day = params.begin
    delta = datetime.timedelta(days=1)

    # Read from the background data and output the simulation
    OUTPUT_IN = f"{params.output_folder}/{params.gas_name}/dataset_extract/"
    OUTPUT_OUT = f"{params.output_folder}/{params.gas_name}/dataset_extract/raw/"
    wavenum = np.arange(params.wave_num_s, params.wave_num_e + 0.25, 0.25)
    print(f"The calculated wavenumber range is {wavenum}\n")
    wavelen = 1.0e7 / wavenum[::-1]     # cm-1 to nm
    wavelen_length = np.shape(wavelen)[0]
    # print(f"The calculated wavelength range is {wavelen}\n")

    # The parallel computation version
    func_params = []
    while day <= params.end:
        time_target = day.strftime("%Y%m%d")
        day += delta
        background_paths = glob.glob(f"{OUTPUT_IN}/dataset_{time_target}*.nc")
        func_params.extend(background_paths)


    print('Suming the dataset...')
    datasets = []
    for f in func_params:
        try:
            ds = xr.open_dataset(f)

            # 检查是否目标变量存在且不是空的
            if 'zenith' in ds and ds.sizes.get('dim_point', 0) > 0:
                ds['thermal_contrast'] = ds['skin_temperature'] - ds['2m_temperature']
                datasets.append(ds)
            else:
                print(f"Skip {f}")
        except Exception as e:
            print(f"Error for {f}: {e}")
    ds = xr.concat(datasets, dim='dim_point') 

    target_var = 'skin_temperature'
    # 定义变量值的区间（分位数或固定间隔）
    num_bins = 20  # 区间数量
    bins = np.linspace(ds[target_var].min(), ds[target_var].max(), num_bins+1)
    counts_ori, bin_edges = np.histogram(ds[target_var], bins=bins)

    print('Sifting the dataset...')
    # 为每个数据点分配区间标签
    labels = pd.cut(ds[target_var].values.flatten(), bins=bins, labels=False)
    ds['bin_label'] = xr.DataArray(labels.reshape(ds[target_var].shape), 
                                dims=ds[target_var].dims)
    # 定义每个区间最大样本数
    max_samples_per_bin = 50_000

    sampled_indices = []
    np.random.seed(42)
    for bin_id in range(num_bins):
        # 获取当前区间的所有数据点索引
        bin_mask = ds['bin_label'] == bin_id
        indices = np.where(bin_mask.values.flatten())[0]
        # 如果样本数超过限制，随机采样
        if len(indices) > max_samples_per_bin:
            indices = np.random.choice(indices, size=max_samples_per_bin, replace=False)
        sampled_indices.extend(indices)
    # 转换为多维索引（假设数据是2D: time x space）
    sampled_indices = np.unravel_index(sampled_indices, ds[target_var].shape)
    # 创建筛选后的数据集
    sampled_ds = ds.isel({dim: sampled_indices[i] for i, dim in enumerate(ds[target_var].dims)})
    print(f"The shape before is {ds[target_var].shape}, and the shape after is {sampled_ds[target_var].shape}")

    # 保存均匀分布的数据集
    output_title = f'{OUTPUT_OUT}/uniform_distribution.nc'
    sampled_ds.to_netcdf(output_title)

    print("Splitting the dataset...")
    # 读取合并后的数据集并拆分成小文件
    num_splits = 120
    split_netcdf(output_title, OUTPUT_OUT, num_splits)
