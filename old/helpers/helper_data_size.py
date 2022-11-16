from data import acic_data, acic_data_test
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset

# continuous outcome variable for training
for i in range(1,17):
    data = acic_data('cont', dgp=i)
    data = to_map_style_dataset(data)
    size = data.__len__()
    print(i, size)

# result:
# 1 100000
# 2 100000
# 3 100000
# 4 200000
# 5 200000
# 6 100000
# 7 200000
# 8 100000
# 9 200000
# 10 200000
# 11 200000
# 12 200000
# 13 200000
# 14 100000
# 15 200000
# 16 200000

# binary outcome variable for training
for name in ['speed', 'epi']:
    data = acic_data('bin', data_name=name)
    data = to_map_style_dataset(data)
    size = data.__len__()
    print(name, size)

# result:
# speed 2400000
# epi 700000

# continuous outcome variable for test
data = acic_data_test('cont')
data = to_map_style_dataset(data)
size = data.__len__()
print('test data cont', size)

# result:
# test data cont 7600

# binary outcome variable for test
data = acic_data_test('bin')
data = to_map_style_dataset(data)
size = data.__len__()
print('test data bin', size)

# result:
# test data bin 8000
