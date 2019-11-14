import requests, zipfile, io
import os 

url = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip' 

data_dir = 'data/'
if not os.path.exists(data_dir):
    print('Creating data diretory..')
    os.makedirs(data_dir)
print('Data directory available..')

print('Downloading data..')
r = requests.get(url)
print(f'request ok: {r.ok}')

print('Unzipping data files')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(data_dir, pwd=b'someone')
