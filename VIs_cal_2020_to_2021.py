
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:37:46 2024

@author: caiwangzheng
"""

import csv
import os
import gdal
import numpy as np
from rasterstats import zonal_stats
os.environ['PROJ_LIB'] = r'C:\Users\caiwangzheng\Anaconda3\envs\deep_regression_keras\Lib\site-packages\pyproj'

shp_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/shp_buffer_join/'
vi_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/vi_tif/'
csv_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/vi_csv/'

# paths = os.walk(r'./test')

paths = os.walk(vi_dir)

for root, dirs, files in paths:
    for d in dirs:
        di = os.path.join(root,d)
        shpi = 'cn_area_'+d+'_join.shp'
        shpi_path = os.path.join(shp_dir, shpi)
        file_names = os.listdir(di)
        
        
        n = 1
        
        row_name = ['Can_ID','Plot_ID']
        for fi in file_names:
            print(fi)
            
            fi_path = os.path.join(di,fi)
            size_metrics = zonal_stats(shpi_path, fi_path, stats = ['mean', 'median', ], geojson_out=True)
            
            m = len(size_metrics)
            
            row_n1 = fi[:-4] + '_mean'
            row_n2 = fi[:-4] + 'median'
            
            
            VIs_t = np.zeros([m,2])
            Can_ID = []
            Plot_ID = []
            row_name.append(row_n1)
            row_name.append(row_n2)
            
            for i in range(0,m):
                VIs_t[i,0] = size_metrics[i]['properties'].get('mean')
                VIs_t[i,1] = size_metrics[i]['properties'].get('median')
                Can_ID.append(size_metrics[i]['properties'].get('Can_ID'))
                Plot_ID.append(size_metrics[i]['properties'].get('Plot_ID'))
            
            if n == 1:
                VIs = VIs_t
                n = 0
            else:
                VIs = np.hstack((VIs, VIs_t))
        
        # export the data csv
        csvFile = open(csv_dir+d+'_size_metrics.csv','wt')
        writer = csv.writer(csvFile, delimiter=',', lineterminator='\n')
        writer.writerow(row_name)
        for i in range(0,m):
            writer.writerow(VIs[i,:])
        
            
            
        
        
                
