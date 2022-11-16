import os
import csv
import pandas as pd

file_names = []
data_source = []
length = []
for i in range(1601, 3201):
    file_name = 'high' + str(i)
    file_names.append(file_name)
    file_path = os.path.join('./data/', file_name + '.CSV')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        row = next(reader)
    dim = len(row)
    length.append(dim)

    if dim == 187:
        data_source.append('speed')
    if dim == 180:
        data_source.append('epi')

d = {'file_name': file_names, 'data_source': data_source}
temp = pd.DataFrame(d)
temp.to_csv('./binary_data_source.csv', index=False)






