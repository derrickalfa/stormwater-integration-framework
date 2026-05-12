# GIS Spatial Attribution of Predictors to Manholes

**Tool**: ArcMap 10.x  
**Coordinate system**: Hong Kong 1980 Grid (EPSG:2326)

This step produces `stromwatermnaholes_connections_with_data.xls`, which is read by Script 06.

---

## Predictor attribution methods

| Predictor | Source format | ArcMap method | Terminology |
|---|---|---|---|
| LULC class | Raster (GeoTIFF, 10 m) | Extract Values to Points (Spatial Analyst) | Raster sampling — nearest neighbour (categorical) |
| Imperviousness (0–100%) | Raster (GeoTIFF, continuous) | Extract Values to Points | Raster sampling — direct (continuous) |
| Population | Vector polygons | Spatial Join — INTERSECT/within | Point-in-polygon attribute transfer |
| Geology | Vector polygons | Spatial Join — INTERSECT/within | Point-in-polygon overlay |
| Traffic stations | Point features | Spatial Join — CLOSEST | Nearest-neighbour station-to-asset |
| Climate stations | Point features | Spatial Join — CLOSEST | Nearest-neighbour station-to-asset |

---

## Imperviousness derivation

Subcatchment imperviousness was derived in two stages:

**Stage A — LULC reclassification**

The 10 m Hong Kong LULC raster was reclassified from categorical class codes
to imperviousness coefficients (%) using the lookup table below.
Source: SWMM Hydrology Reference Manual (Rossman & Huber, 2016);
transport classes from NLCD high-intensity developed land definition.

| LULC Code | Description | Imperviousness (%) |
|---|---|---|
| 1, 2 | Private/Public Residential | 51 |
| 3 | Rural Settlement | 19 |
| 11 | Commercial/Business | 56 |
| 21–23 | Industrial/Warehouses | 76 |
| 31 | Government/Institutional | 34 |
| 32 | Open Space/Recreation | 11 |
| 41–44 | Roads/Railways/Airport/Port | 95 |
| 51 | Cemeteries | 11 |
| 52 | Utilities | 55* |
| 53 | Vacant/Construction | 11 |
| 61 | Agricultural Land | 2 |
| 71–73 | Woodland/Shrubland/Grassland | 2 |
| 62, 74, 81, 83, 91, 92 | Water/Barren/Rocky | 0 |

*Utilities coefficient = mean of Institutional (34%) and Industrial (76%).

ArcMap tool: Reclassify (Spatial Analyst)

**Stage B — Subcatchment area-weighted average**

The reclassified imperviousness raster was overlaid with the contributing
subcatchment polygon for each manhole. The area-weighted mean imperviousness
within each subcatchment was computed and stored as `Imperv_Point`.

---

## References

1. Rossman, L. A., & Huber, W. C. (2016). SWMM Reference Manual Vol. I — Hydrology. U.S. EPA.
2. MRLC Consortium (2023). National Land Cover Database Class Legend.
3. Feng, B., Zhang, Y., & Bourke, R. (2021). Natural Hazards, 106(1), 613–627.
4. Li, J., & Bortolot, Z. J. (2022). Journal of Cleaner Production, 344, 130992.
