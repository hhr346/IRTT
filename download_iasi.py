'''
Download L1C and L2 with nc
'''
import time
import eumdac
import fnmatch
import shutil
import os
import datetime

# The function for the datatailor download
def download(product, output_path):

    chain = eumdac.tailor_models.Chain(
        id=f'{product}',
        product=product_name,
        format='netcdf4_satellite',
    )
    customisation = datatailor.new_customisation(product, chain)
    status = customisation.status

    # Customisation Loop
    while status:
        # Get the status of the ongoing customisation
        try:
            status = customisation.status
        except Exception as error:
            print(error)
            time.sleep(5)
            pass

        if "DONE" in status:
            print(f"Customisation {customisation._id} is successfully completed.")
            break
        elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
            print(f"Customisation {customisation._id} was unsuccessful. Customisation log is printed.\n")
            print(customisation.logfile)
            break
        elif "QUEUED" in status:
           print(f"Customisation {customisation._id} is queued.")
        elif "RUNNING" in status:
           print(f"Customisation {customisation._id} is running.")
           time.sleep(2)

    try:
        if "DONE" in status:
            # Delete the job if the file already exists
            nc, = fnmatch.filter(customisation.outputs, '*.nc')
            output_name = output_path + str(product) + '.nc'
            # if os.path.exists(f'{nc}'):
            if os.path.exists(f'{output_name}'):
                pass

            # Or download the file
            else:
                jobID = customisation._id
                print(f'Time is {datetime.datetime.now()}')
                print(f"Downloading the nc output of the customisation {jobID}")
                with customisation.stream_output(nc, ) as stream, open(output_name, mode='wb') as fdst:
                    # open(stream.name, mode='wb') as fdst:
                    shutil.copyfileobj(stream, fdst)
                print(f"Dowloaded the nc output of the customisation {jobID}")
    except Exception as error:
        pass
    # customisation.delete()


# Insert your personal key and secret into the single quotes
consumer_key = 'xxx'
consumer_secret = 'xxx'

credentials = (consumer_key, consumer_secret)
token = eumdac.AccessToken(credentials)
try:
    print(f"This token '{token}' expires {token.expiration}")
except Exception as error:
    print(f"Error when tryng the request to the server: '{error}'")

# First choose which collection to work with
datastore = eumdac.DataStore(token)
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

print(f'Downloading {start}-{end} of status {status} for {satellite_type}')

if status == 0:
    collection_name = 'EO:EUM:DAT:METOP:IASIL1C-ALL' # L1C product
    output_path = '/exports/d5/hhr346/' + satellite_type + '/LEVEL1/'
    product_name = 'IASIL1'
elif status == 1:
    collection_name = 'EO:EUM:DAT:0758'           # PCS product, not working
    output_path = '/exports/d5/hhr346/' + satellite_type + '/LEVEL1_PCS/'
    product_name = 'IASIL1'
elif status == 2:
    collection_name = 'EO:EUM:DAT:METOP:IASSND02' # L2 product
    output_path = '/exports/d5/hhr346/' + satellite_type + '/LEVEL2/'
    product_name = 'IASISND02'
selected_collection = datastore.get_collection(collection_name)


# Retrieve datasets that match our filter
products = selected_collection.search(
    dtstart=start,
    dtend=end,
    sat=satellite_type,
    sort="start,time,1",
    )
try:
    print(f'Found {products.total_results} datasets for the given time range and geometry:')
except Exception as error:
    print(f"Unexpected error: {error}")

# Download all products
print("Start to download...")
datatailor = eumdac.DataTailor(token)
print(datatailor.quota)

for product in products:
    print(product)
    if datetime.datetime.now() > token.expiration:
        print(f"Now is {datetime.datetime.now()}, late for {token.expiration}. Change the token!")
        token = eumdac.AccessToken(credentials)
        datatailor = eumdac.DataTailor(token)
        print(datatailor.quota)
        print(f"This token '{token}' expires {token.expiration}")
    try:
        download(product, output_path)
    except Exception as error:
        print(f"Unexpected error: {error}")
print('All downloads are finished.')
