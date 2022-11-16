import os.path
import pandas as pd

root_dir = './TestDatasets_highD'
for i in range(8):
    file_name_cf = 'highDim_testdataset' + str(i+1) + '_cf.csv'
    file_name_data = 'highDim_testdataset' + str(i+1) + '.csv'
    cf = pd.read_csv(os.path.join(root_dir, file_name_cf), header=None, delimiter=',')
    data = pd.read_csv(os.path.join(root_dir, file_name_data), header=None, delimiter=',')

    if i in [0, 1, 4, 5]:
        data_type = 'bin'
    else:
        data_type = 'cont'

    temp = pd.concat([cf, data], axis=1)
    temp_file_name = data_type + str(i) + '.csv'
    temp.to_csv(os.path.join('./test_data', temp_file_name), header = None, index=False)
