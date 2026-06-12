import cdsapi
import datetime
import time
import subprocess

def download_with_retry(client, dataset, request, filename, max_retries=10):
    attempt = 0
    result = client.retrieve(dataset, request)
    print(result)
    download_link = result.location
    print(download_link)
    while attempt < max_retries:
        try:
            # 用 wget 下载，支持断点续传（-c）
            cmd = f"wget -c {download_link} -O {filename}"
            cmd = cmd.replace(":443", "", 1)

            # subprocess.run(cmd, shell=True, check=True)
            # print(f"Successfully downloaded {filename}")
            # time.sleep(30)

            with open(f"{filename}.log", "a") as log:
                subprocess.Popen(cmd, shell=True, stdout=log)
                print(f"Successfully downloaded {filename}")
            time.sleep(6000)
            return
        except Exception as e:
            attempt += 1
            print(f"Error: {e} (attempt {attempt}/{max_retries}). Retrying...")
            time.sleep(30)
    print(f"Failed to download {filename} after {max_retries} attempts.")


if __name__ == "__main__":
    begin = datetime.date(2022, 1, 1)
    end = datetime.date(2022, 1, 1)
    delta = datetime.timedelta(days=1)
    d = begin

    while d <= end:
        year = d.strftime("%Y")
        month = d.strftime("%m")
        day = d.strftime("%d")
        file_target = f"{year}/profile_{year}{month}{day}.nc"
        d += delta

        print(f"Downloading data for {year}-{month}-{day}")
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
        client = cdsapi.Client()
        download_with_retry(client, dataset, request, file_target)
