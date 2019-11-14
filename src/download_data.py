import requests, zipfile, ioi

url = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip' 

data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedir(data_dir)
r = requests.get(url)
print(f'request ok: {r.ok}')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(data_dir)
