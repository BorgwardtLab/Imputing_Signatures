import requests, zipfile, io
import os 

# Download UCR:
url = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip' 

data_dir = 'data/'
UCR_dir = data_dir + 'UCRArchive_2018'
UEA_dir = data_dir + 'Multivariate_arff'
if not os.path.exists(UCR_dir): # check if UCR directory exists, if not download the UCR archive
    if not os.path.exists(data_dir):
        print('Creating data diretory..')
        os.makedirs(data_dir)
    print('Data directory available..')

    print('Downloading UCR data..')
    r = requests.get(url)
    print(f'request ok: {r.ok}')

    print('Unzipping data files')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir, pwd=b'someone')

# Download UEA (multi-variate):
if not os.path.exists(UEA_dir): # check if UEA directory exists, if not download the UEA archive
    url = 'https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip' 
    print('Downloading UEA data..')
    r = requests.get(url)
    print(f'request ok: {r.ok}')

    print('Unzipping data files')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)


