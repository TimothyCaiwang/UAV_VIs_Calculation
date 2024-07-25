# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:44:52 2024

@author: caiwangzheng
"""

import csv
import os
import gdal
import numpy as np
os.environ['PROJ_LIB'] = r'C:\Users\caiwangzheng\Anaconda3\envs\deep_regression_keras\Lib\site-packages\pyproj'

img_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/ortho_samples/'

# shp_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/'

vi_dir = 'E:/Caiwang_ZHENG/Strawberry_Dry_Biomass_Prediction_2021_To_2022/data_2020_to_2021/train_samples_images/UTM/vi_tif/'

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
        #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i,:,:])
    del dataset


file_names = os.listdir(img_dir)

for fi in file_names:
    if(fi.endswith('.tif')):
        print(fi)
        fi_path = os.path.join(img_dir, fi)
        # SHPi = shp_dir+fi[0:8]+'_contours_'+'join.shp' # cn_area_20210303_UTM.shp
        
        image = gdal.Open(fi_path)
        Ri = image.GetRasterBand(1).ReadAsArray()
        Gi = image.GetRasterBand(2).ReadAsArray()
        Bi = image.GetRasterBand(3).ReadAsArray()
        NIRi = image.GetRasterBand(4).ReadAsArray()
        RedEdgei = image.GetRasterBand(5).ReadAsArray()
        
        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        
        ARI1 = 1/Gi - 1/RedEdgei # Generate ARI1 Tiff (Anthocyanin Reflectance Index 1)
        ARI2 = (1/Gi - 1/RedEdgei) * NIRi # Anthocyanin Reflectance Index 2 (ARI2)
        CIG = NIRi/Gi - 1 # Chlorophyll Index Green (CIG)
        CIRE = NIRi/RedEdgei - 1 # Chlorophyll Index Red Edge (CIRE)
        CIVE = 0.441*Ri - 0.811*Gi + 0.385*Bi + 18.78745 # (https://www.redalyc.org/journal/4457/445758367004/html/)
        CRI1 = 1/Bi - 1/Gi # Carotenoid Reflectance Index 1 (CRI1)
        CRI2 = 1/Bi -1/RedEdgei # Carotenoid Reflectance Index 2 (CRI2)
        CRIG = (1/Bi - 1/Gi) * NIRi # Carotenoid Reflectance Index Green (CRIG)
        CRIRE = (1/Bi - 1/RedEdgei)*NIRi # Carotenoid Reflectance Index Red Edge
        ENDVI = (NIRi + Gi - 2*Bi)/(NIRi + Gi + 2*Bi) # Enhanced Normalized Difference Vegetation Index
        EVI = 2.5*((NIRi - Ri)/((NIRi + 6 * Ri - 7.5 * Bi) + 1)) # (Enhanced Vegetation Index)
        EXG = 2*Gi - Ri - Bi #  (https://plantmethods.biomedcentral.com/articles/10.1186/s13007-019-0402-3/tables/6)
        GDVI = (NIRi - Gi) # Generalized Difference Vegetation Index
        GLI = (2 * Gi - Ri - Bi)/(2 * Gi + Ri + Bi) # Green leaf Index
        GNDVI = (NIRi - Gi)/(NIRi + Gi) # Green Normalized Difference Vegetation Index
        GSAVI = (NIRi - Gi)/(NIRi + Gi + 0.5)*1.5 # Green Soil Adjusted Vegetation Index       
        LCI = (NIRi - RedEdgei)/(NIRi + Ri) # Leaf Chlorophyll Index
        MCARI1 = 1.2 * (2.5 * (NIRi - Ri) - 1.3 * (NIRi - Gi)) # Modified Chlorophyll Absorption Ratio Index
        MSR = (NIRi - Bi)/(Ri - Bi) # Modified Simple Ratio
        NDRE = (NIRi - RedEdgei)/(NIRi + RedEdgei)
        NDVI = (NIRi - Ri)/(NIRi + Ri)
        NGRDI = (Gi - Ri)/(Gi + Ri) # Normalized Difference Green/Red Normalized green red difference index)
        NPCI = (Ri - Bi)/(Ri + Bi) # Normalized Pigment Chlorophlyll Index
        OSAVI = (NIRi - Ri)/(NIRi + Ri + 0.16)*(1 + 0.16)
        PSND2 = (NIRi - Bi)/(NIRi + Bi) # Pigment Specific Normalized Difference 2
        PSRI = (Ri - Bi)/RedEdgei # Plant Senescence Reflectance Index
        # RGRI
        MRVI = NIRi/Ri # RVI
        CCCI = (NIRi - RedEdgei) * (NIRi + Ri)/(NIRi + RedEdgei)/(NIRi - Ri)
        VARI = (Gi - Ri)/(Gi + Ri - Bi)
        
        vi_dir_i = vi_dir + fi[0:8] + '/'
        os.makedirs(vi_dir_i) 
        
        R_tif = vi_dir_i + 'R.tif'
        G_tif = vi_dir_i + 'G.tif'
        B_tif = vi_dir_i + 'B.tif'
        NIR_tif = vi_dir_i + 'NIR.tif'
        RedEdge_tif = vi_dir_i + 'RedEdge.tif'
        ARI1_tif = vi_dir_i + 'ARI1.tif'
        ARI2_tif = vi_dir_i + 'ARI2.tif'
        CIG_tif = vi_dir_i + 'CIG.tif'
        CIRE_tif = vi_dir_i + 'CIRE.tif'
        CIVE_tif = vi_dir_i + 'CIVE.tif'
        CRI1_tif = vi_dir_i + 'CRI1.tif'
        CRI2_tif = vi_dir_i + 'CRI2.tif'
        CRIG_tif = vi_dir_i + 'CRIG.tif'
        CRIRE_tif = vi_dir_i + 'CRIRE.tif'
        ENDVI_tif = vi_dir_i + 'ENDVI.tif'
        EVI_tif = vi_dir_i + 'EVI.tif'
        EXG_tif = vi_dir_i + 'EXG.tif'
        GDVI_tif = vi_dir_i + 'GDVI.tif'
        GLI_tif = vi_dir_i + 'GLI.tif'
        GNDVI_tif = vi_dir_i + 'GNDVI.tif'
        GSAVI_tif = vi_dir_i + 'GSAVI.tif'
        LCI_tif = vi_dir_i + 'LCI.tif'
        MCARI1_tif = vi_dir_i + 'MCARI1.tif'
        MSR_tif = vi_dir_i + 'MSR.tif'
        NDRE_tif = vi_dir_i + 'NDRE.tif'
        NDVI_tif = vi_dir_i + 'NDVI.tif'
        NGRDI_tif = vi_dir_i + 'NGRDI.tif'
        NPCI_tif = vi_dir_i + 'NPCI.tif'
        OSAVI_tif = vi_dir_i + 'OSAVI.tif'
        PSND2_tif = vi_dir_i + 'PSND2.tif'
        PSRI_tif = vi_dir_i + 'PSRI.tif'
        MRVI_tif = vi_dir_i + 'MRVI.tif'
        CCCI_tif = vi_dir_i + 'CCCI.tif'
        VARI_tif = vi_dir_i + 'VARI.tif'
        
        writeTiff(Ri, img_geotrans, img_proj, R_tif)
        writeTiff(Gi, img_geotrans, img_proj, G_tif)
        writeTiff(Bi, img_geotrans, img_proj, B_tif)
        writeTiff(NIRi, img_geotrans, img_proj, NIR_tif)
        writeTiff(RedEdgei, img_geotrans, img_proj, RedEdge_tif)
        writeTiff(ARI1, img_geotrans, img_proj, ARI1_tif)
        writeTiff(ARI2, img_geotrans, img_proj, ARI2_tif)
        writeTiff(CIG, img_geotrans, img_proj, CIG_tif)
        writeTiff(CIRE, img_geotrans, img_proj, CIRE_tif)
        writeTiff(CIVE, img_geotrans, img_proj, CIVE_tif)
        writeTiff(CRI1, img_geotrans, img_proj, CRI1_tif)
        writeTiff(CRI2, img_geotrans, img_proj, CRI2_tif)
        writeTiff(CRIG, img_geotrans, img_proj, CRIG_tif)
        writeTiff(CRIRE, img_geotrans, img_proj, CRIRE_tif)
        writeTiff(ENDVI, img_geotrans, img_proj, ENDVI_tif)
        writeTiff(EVI, img_geotrans, img_proj, EVI_tif)
        writeTiff(EXG, img_geotrans, img_proj, EXG_tif)
        writeTiff(GDVI, img_geotrans, img_proj, GDVI_tif)
        writeTiff(GLI, img_geotrans, img_proj, GLI_tif)   
        writeTiff(GNDVI, img_geotrans, img_proj, GNDVI_tif)
        writeTiff(GSAVI, img_geotrans, img_proj, GSAVI_tif)
        writeTiff(LCI, img_geotrans, img_proj, LCI_tif)
        writeTiff(MCARI1, img_geotrans, img_proj, MCARI1_tif)                 
        writeTiff(MSR, img_geotrans, img_proj, MSR_tif)
        writeTiff(NDRE, img_geotrans, img_proj, NDRE_tif)
        writeTiff(NDVI, img_geotrans, img_proj, NDVI_tif)
        writeTiff(NGRDI, img_geotrans, img_proj, NGRDI_tif)
        writeTiff(NPCI, img_geotrans, img_proj, NPCI_tif)
        writeTiff(OSAVI, img_geotrans, img_proj, OSAVI_tif)
        writeTiff(PSND2, img_geotrans, img_proj, PSND2_tif)
        writeTiff(PSRI, img_geotrans, img_proj, PSRI_tif)
        writeTiff(MRVI, img_geotrans, img_proj, MRVI_tif)
        writeTiff(CCCI, img_geotrans, img_proj, CCCI_tif)
        writeTiff(VARI, img_geotrans, img_proj, VARI_tif)
         
        


     
