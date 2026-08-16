"""
Eval the model with all the orbits
Write the output data
Denomalize the output and calculate the column
"""
import netCDF4 as nc
import glob
import numpy as np
import datetime
import torch
from train_p import EncoderRegressor, apply_normalizer
# from train_temp_noise import EncoderRegressor, apply_normalizer
import gc
gc.collect()  # 强制垃圾回收


def denormalize(data, min, max, scale=1, log=False):
    data = (data + 1) / 2
    data = data * (max - min) + min 
    if log:
        data = np.exp(data)
    data = data / scale
    return data


def readInput(filepath):
    print(f"Reading file {filepath}")
    try:
        with nc.Dataset(filepath, mode='r') as ncfile:
            input_data = np.asarray(ncfile.variables['input_data'][:, :, :n_features])
    except Exception as error:
        print('Reading inputData Error', error)

    # Sift the negative ones
    neg_mask = input_data < 0
    channel_has_neg = neg_mask.any(axis=2)
    print(f"There are {np.sum(channel_has_neg)} samples with negative input data.")
    input_data[channel_has_neg, :] = np.nan

    # Do the normalization
    normalizer = np.load(normalizer_path)
    mean, std = normalizer['mean'], normalizer['std']
    input_data = apply_normalizer(input_data, mean, std)
    return input_data


def eval_with_uncertainty(input_tensor):
    model.eval()
    outputs = []
    outputs_uncertainty = []
    time_now = datetime.datetime.now()
    for i in range(0, len(input_tensor), eval_batch_size):
        batch = input_tensor[i:i + eval_batch_size]
        with torch.no_grad():
            mu, log_var = model(batch)
        outputs.append(mu)
        outputs_uncertainty.append(log_var.exp().sqrt())

    time_then = datetime.datetime.now()
    print(f"Evaluation time: {time_then - time_now}, Per sample time: {(time_then - time_now)/len(input_tensor)}")
    output_data = torch.cat(outputs, dim=0).cpu().numpy()
    output_data_uncertainty = torch.cat(outputs_uncertainty, dim=0).cpu().numpy()
    return output_data, output_data_uncertainty


def process_func(level1_file):
    level1_file = level1_file[0]
    try:
        input_data = readInput(level1_file)
        # Reshape the input data to fit the model
        shape_input = input_data.shape
        input_data = input_data.reshape(-1, shape_input[2])
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

        ratio, uncertainty = eval_with_uncertainty(input_tensor)

        # 将输出数据转换为原始形状
        ratio = ratio.reshape(shape_input[0], shape_input[1])
        normalizer = np.load(normalizer_path)
        label_min, label_max = normalizer['label_min'], normalizer['label_max']
        ratio_denormalize = denormalize(ratio,  label_min, label_max, 1E-18)
        column = ratio_denormalize

        uncertainty = uncertainty.reshape(shape_input[0], shape_input[1])
        uncertainty_denormalize = denormalize(ratio+uncertainty, label_min, label_max, 1E-18) - denormalize(ratio, label_min, label_max, 1E-18)
        column_uncertainty = uncertainty_denormalize

        # Save to the nc file
        with nc.Dataset(level1_file, 'a', format='NETCDF4') as dataset:
            variable1 = dataset.variables['output_data']
            variable2 = dataset.variables['column']
            try:
                dataset.createVariable('uncertainty', 'f4', ('dim1', 'dim2'))
            except Exception:
                pass
            variable3 = dataset.variables['uncertainty']

            variable1[:] = ratio
            variable2[:] = column
            variable3[:] = column_uncertainty

    except Exception as error:
        print('\033[0;31mProcessing error! %s\033[0m' %error)
        return None


n_features = 154
n_layer = 4
n_head = 8
batch_size = 128
n_embd = 256         # embedding dimension
dropout = 0.1
weight_decay = 1e-2
eval_batch_size = 2 ** 11
suffix = '_final_p_300'
# suffix = '_final_t_300'
gas_name = 'CO'

begin = datetime.date(2022, 1, 1)
end = datetime.date(2022, 1, 1)

model_path = f'./model_path/{gas_name}_IRTT_model{suffix}.pth'
normalizer_path = f'./model_path/{gas_name}_IRTT_normalizer{suffix}.npz'
print(f"\033[33mLoading model from {model_path}\033[0m")
day = begin
delta = datetime.timedelta(days=1)
device = torch.device("cuda:0")

model = EncoderRegressor(seq_len=n_features, n_features=n_features,
                            n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()

while day <= end:
    time_target = day.strftime("%Y%m%d")
    day += delta
    level1_files = sorted(glob.glob(f"/your_path_here/{gas_name}_IASI_xxx_1C_M01_{time_target}*"))
    for level1_file in level1_files:
        process_func([level1_file])
