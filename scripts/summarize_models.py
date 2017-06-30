import os
import pickle
from utils.vehicle_functions import *

files = [f for f in os.listdir('models') if f.endswith('.p')]
with open('hyper-train-results.csv', 'w') as output:
    header = 'filename, test_acc, color_space, hog_orient, hog_pix_per_cell,'\
             'hog_cell_per_block, hog_channels, spatial_size, hist_bins\n'
    output.write(header)
    for filename in files:
        with open(os.path.join('models', filename), 'rb') as model:
            obj = pickle.load(model)
            row = ','.join(['{}']*(header.count(',')+1))
            row = row.format(filename, obj['acc'], obj['options'].color_space,
                             obj['options'].orient, obj['options'].pix_per_cell,
                             obj['options'].cell_per_block, obj['options'].hog_channel,
                             obj['options'].spatial_size[0], obj['options'].hist_bins)
            output.write(row + '\n')
