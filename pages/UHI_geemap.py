import streamlit as st
import geemap.foliumap as geemap
import ee

# Initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Create a Streamlit app
st.title("Urban Heat Island (UHI) Mapping in Mombasa")

# Define the region of interest (Mombasa - Mvita)
admin = ee.FeatureCollection("projects/ee-levice/assets/constituencies")
mvita = admin.filter(ee.Filter.eq('CONSTITUEN', 'MVITA'))
geometry = mvita.geometry()

# Define Cloud Masking Function
def cloud_mask(image):
    scored = ee.Algorithms.Landsat.simpleCloudScore(image)
    mask = scored.select(['cloud']).lte(10)
    return image.updateMask(mask)

# Filter Landsat 8 Data
landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
    .filter(ee.Filter.date('2014-01-01', '2024-10-01')) \
    .filter(ee.Filter.bounds(geometry)) #\
    #.map(cloud_mask)

# Compute Median Composite
median = landsat.median().clip(geometry)

# Compute NDVI
ndvi = median.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
# Create Map
Map = geemap.Map(center=[-4.05, 39.67], zoom=12)
Map.centerObject(geometry, 12)
# Add Layers to Map
ndvi_vis = {
    'min': -1,
    'max': 1,
    'palette': [
  'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
  '74A901', '66A000', '529400', '3E8601', '207401', '056201',
  '004C00', '023B01', '012E01', '011D01', '011301']
  }
Map.addLayer(ndvi,ndvi_vis, "NDVI")

# Find the minimum and maximum NDVI values
min_max = ndvi.reduceRegion(
    reducer=ee.Reducer.min().combine(
        reducer2=ee.Reducer.max(),
        sharedInputs=True
    ),
    geometry=geometry,
    scale=30,
    maxPixels=1e9
)

# Extract min and max values
ndvi_min = ee.Number(min_max.get('NDVI_min'))
ndvi_max = ee.Number(min_max.get('NDVI_max'))

# Compute Fractional Vegetation Cover (FV)
fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).rename('FV')

# Add FV band to NDVI image
with_fv = ndvi.addBands(fv)

# Define FV visualization parameters
fv_vis = {
    'min': 0,
    'max': 0.7,
    'palette': [
        'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
        '74A901', '66A000', '529400', '3E8601', '207401', '056201',
        '004C00', '023B01', '012E01', '011D01', '011301'
    ]
}

# Create a map
Map = geemap.Map()
Map.centerObject(geometry, 12)
Map.addLayer(fv, fv_vis, "Fractional Vegetation (FV)")

# Emissivity calculations
a = ee.Number(0.004)
b = ee.Number(0.986)
not_water = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('occurrence').mask().Not()
em = fv.multiply(a).add(b).rename('EMM').updateMask(not_water)

# Define Emissivity visualization parameters
em_vis = {
    'min': 0.98,
    'max': 0.99,
    'palette': ['blue', 'white', 'green']
}

Map.addLayer(em, em_vis, "Emissivity (EMM)")

# Load Landsat thermal image (replace with actual image)
thermal = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA") \
    .filterBounds(geometry) \
    .filterDate('2014-01-01', '2024-10-01') \
    .median().select('B10').clip(geometry)

# Compute Land Surface Temperature (LST)
lst_landsat = thermal.expression(
    '(Tb / (1 + (0.001145 * (Tb / 1.438)) * log(Ep))) - 273.15',
    {
        'Tb': thermal.select('B10'),
        'Ep': em.select('EMM')
    }
).updateMask(not_water).rename('LST')

# Define LST visualization parameters
lst_vis = {
    'min': 25,
    'max': 35,
    'palette': ['blue', 'white', 'red']
}

Map.addLayer(lst_landsat, lst_vis, "Land Surface Temperature (LST)")

# Display map in Streamlit
Map.to_streamlit(height=600)