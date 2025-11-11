# PAZ Processing Graph Summary

## 1. Read
Purpose: Ingest original PAZ Stripmap SSC Level-1B product.
Parameters:
- File: `../PAZ/SAR_SSC/PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536/PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536.xml`
- copyMetadata: true
- useAdvancedOptions: false
Output: Source product with original metadata.

## 2. Calibration
Purpose: Radiometric calibration to physical backscatter (creating Sigma0; Beta0 internal).
Parameters:
- outputSigmaBand: true (Sigma0 band generated)
- createBetaBand: true (Beta0 computed internally)
- outputBetaBand: false (Beta0 not written)
- createGammaBand / outputGammaBand: false (no Gamma0)
- outputImageScaleInDb: false (linear scale retained)
- outputImageInComplex: false (intensity image; source already detected)
- auxFile: Latest Auxiliary File
Result: Calibrated Sigma0 intensity (HH) retained; flags set (abs_calibration_flag=1).

## 3. Terrain-Flattening
Purpose: Radiometric terrain normalization (correct incidence/layover-induced radiometric distortions) using external high‑resolution DEM.
Parameters:
- demName: External DEM
- externalDEMFile: `DEM/TRUE_1.5m_IGN_mb_lidar/DEM__EPSG_32632__1_5m.tif`
- externalDEMNoDataValue: -999.0
- externalDEMApplyEGM: true (EGM geoid applied)
- demResamplingMethod: CUBIC_CONVOLUTION
- nodataValueAtSea: true
- outputSigma0: false (Sigma0 already from Calibration)
- outputSimulatedImage: false
- oversamplingMultiple: 1.0
- additionalOverlap: 0.1
Result: Flattened radiometry (range_spread_comp_flag=1, ant_elev_corr_flag=1) before orthorectification.

## 4. Write (Intermediate)
Purpose: Persist intermediate calibrated + terrain‑flattened product for traceability / later reuse.
Parameters:
- file: `output/Cal_Tf_Tc/PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536/PAZ1_SAR__SSC______SM_S_SRA_20211122T172529_20211122T172536_Cal_Tf_Tc.dim`
- formatName: BEAM-DIMAP
- deleteOutputOnFailure: true
- writeEntireTileRows: false
- clearCacheAfterRowWrite: false
Result: BEAM-DIMAP intermediate dataset.

## 5. Terrain-Correction (Range-Doppler Orthorectification)
Purpose: Geometric correction (project to map geometry) + resampling to uniform pixel spacing.
Known / derived parameters:
- mapProjection: Geographic WGS84 ("WGS84(DD)")
- pixelSpacingInMeter: 3.0 (target ground spacing)
- alignToStandardGrid: false
- nodataValueAtSea: true
- DEM: `/home/test/Téléchargements/DEM__EPSG_32632__1_5m.tif`
- DEM resampling: CUBIC_CONVOLUTION (from metadata field)
- Output geocoded resolution: ~2.69494585e-5 deg (lat & lon) ≈ 3 m at scene latitude
- is_terrain_corrected: 1 (flag confirms)
Result: Final geocoded, terrain-corrected Sigma0 HH image in geographic coordinates.

## Functional Flow
1. Read raw SSC L1B (detected) PAZ Stripmap product.
2. Radiometric calibration (Sigma0) retaining linear scale.
3. Radiometric terrain normalization using external 1.5 m DEM (improves radiometric consistency over relief).
4. Intermediate write (checkpoint).
5. Range-Doppler orthorectification to WGS84 geographic grid at ~3 m spacing.

## Notes
- No multilooking (multilook_flag=0); native ~3 m spacing preserved.
- Product polarization: HH only.
- Ascending pass (PASS=ASCENDING), right-looking geometry.
- Gamma0 not produced; Beta0 not exported.
- Geographic (lat/lon) output chosen instead of projected UTM (may introduce non-uniform metric distortion with latitude; acceptable for small area or quick-look—consider UTM for strict metric analyses).
- Doppler centroid polynomial coefficients provided per time slice (used internally in geometric model).
- **Checkpoint utilization**: The intermediate write checkpoint is essential for subsequent terrain correction with mask creation (layover, shadow or both) using SRTM DEM. This is necessary because the homemade high-resolution DEM is too small compared to the full image extent, causing mask generation to fail without the SRTM fallback.

---

# Metadata Summary Table

| Satellite | Heure | Date | Pass | Polarisations | Angle Incidence | Résolution Range | Résolution Azimuth |
|-----------|--------|------|------|---------------|------------------|------------------|--------------------|
| TerraSAR-X | 05:44:10 | 2007-10-24 | Descending | HH/VV | 37.8° | 1.9 m | 6.0 m |
| TerraSAR-X | 05:44:09 | 2007-11-04 | Descending | HH | 37.4° | 2.9 m | 3.0 m |
| TerraSAR-X | 05:44:05 | 2008-01-09 | Descending | HH | 37.3° | 2.9 m | 3.0 m |
| TerraSAR-X | 17:25:01 | 2008-01-11 | Ascending | HH | 44.4° | 2.5 m | 3.0 m |
| TerraSAR-X | 05:44:15 | 2008-08-05 | Descending | HH/VV | 38.2° | 1.9 m | 3.2 m |
| TerraSAR-X | 05:44:16 | 2008-08-16 | Descending | HH/VV | 37.5° | 1.9 m | 3.2 m |
| TerraSAR-X | 05:44:16 | 2008-08-16 | Descending | HH/VV | 37.5° | 1.9 m | 3.2 m |
| TerraSAR-X | 05:44:16 | 2008-09-29 | Descending | HH | 37.2° | 2.9 m | 3.3 m |
| TerraSAR-X | 05:44:15 | 2008-10-10 | Descending | HH | 37.2° | 2.9 m | 3.3 m |
| TerraSAR-X | 05:44:16 | 2008-10-21 | Descending | HH | 37.2° | 2.9 m | 3.3 m |
| TerraSAR-X | 05:44:11 | 2009-01-06 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:10 | 2009-01-17 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:10 | 2009-01-28 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:10 | 2009-02-08 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:10 | 2009-02-19 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:10 | 2009-03-02 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:11 | 2009-03-13 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:11 | 2009-03-24 | Descending | HH/HV | 37.9° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:14 | 2009-05-29 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:10 | 2009-05-31 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:14 | 2009-06-09 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:10 | 2009-06-11 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:15 | 2009-06-20 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:10 | 2009-06-22 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:15 | 2009-07-01 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:11 | 2009-07-03 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:16 | 2009-07-12 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:12 | 2009-07-14 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:12 | 2009-07-25 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:14 | 2009-08-05 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:18 | 2009-08-14 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:14 | 2009-08-16 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:18 | 2009-08-25 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| TerraSAR-X | 17:25:14 | 2009-08-27 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:15 | 2009-09-18 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:16 | 2009-09-29 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:16 | 2009-10-10 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:16 | 2009-10-21 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:19 | 2010-04-13 | Descending | HH/HV/VH/VV | 37.8° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:20 | 2010-04-24 | Descending | HH/HV/VH/VV | 37.8° | 1.9 m | 6.6 m |
| TerraSAR-X | 05:44:19 | 2010-05-05 | Descending | HH/HV/VH/VV | 37.8° | 1.9 m | 6.6 m |
| TanDEM-X | 17:25:18 | 2011-05-05 | Ascending | HH | 44.4° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:19 | 2011-05-16 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:20 | 2011-05-27 | Ascending | HH | 44.4° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:20 | 2011-06-07 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:21 | 2011-06-18 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:21 | 2011-06-29 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:22 | 2011-07-10 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:23 | 2011-07-21 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:23 | 2011-08-01 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:24 | 2011-08-12 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 17:25:25 | 2011-08-23 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:29 | 2011-09-01 | Descending | VV | 37.2° | 2.9 m | 3.3 m |
| TanDEM-X | 17:25:25 | 2011-09-03 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TerraSAR-X | 05:44:30 | 2011-09-12 | Descending | VV | 37.3° | 2.9 m | 3.3 m |
| TerraSAR-X | 05:44:30 | 2011-09-23 | Descending | VV | 37.2° | 2.9 m | 3.3 m |
| TanDEM-X | 17:25:26 | 2011-09-25 | Ascending | HH | 44.5° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:39 | 2013-10-23 | Ascending | HH | 44.4° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:38 | 2013-10-23 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:37 | 2013-11-03 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:39 | 2013-11-03 | Ascending | HH | 44.4° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:38 | 2013-11-14 | Ascending | HH | 44.4° | 2.5 m | 3.3 m |
| TanDEM-X | 17:25:37 | 2013-11-14 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:21 | 2020-01-08 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:17 | 2020-01-10 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 17:25:16 | 2020-01-21 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:21 | 2020-01-30 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:20 | 2020-02-10 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:16 | 2020-02-12 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:20 | 2020-02-21 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:16 | 2020-02-23 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:20 | 2020-03-03 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:16 | 2020-03-05 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 17:25:17 | 2020-03-16 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:21 | 2020-03-25 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:17 | 2020-03-27 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:22 | 2020-04-05 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:22 | 2020-04-16 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:18 | 2020-04-18 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:23 | 2020-04-27 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:23 | 2020-05-08 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:19 | 2020-05-10 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 17:25:20 | 2020-05-21 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:24 | 2020-05-30 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:20 | 2020-06-01 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:25 | 2020-06-10 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:20 | 2020-06-12 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:26 | 2020-06-21 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:22 | 2020-06-23 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:25 | 2020-07-02 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:21 | 2020-07-04 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:26 | 2020-07-13 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:22 | 2020-07-15 | Ascending | HH/HV | 44.9° | 1.7 m | 6.6 m |
| PAZ | 05:44:27 | 2020-07-24 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:23 | 2020-07-26 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:27 | 2020-08-04 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:23 | 2020-08-06 | Ascending | HH/HV | 44.9° | 1.7 m | 6.6 m |
| PAZ | 05:44:27 | 2020-08-15 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:29 | 2020-08-26 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:25 | 2020-08-28 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:29 | 2020-09-06 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:26 | 2020-09-19 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:31 | 2020-09-28 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:26 | 2020-09-30 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:32 | 2020-10-09 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 17:25:28 | 2020-10-11 | Ascending | HH/HV | 44.8° | 1.7 m | 6.6 m |
| PAZ | 05:44:31 | 2020-10-20 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:31 | 2020-10-31 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:32 | 2020-11-11 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:31 | 2020-11-22 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:31 | 2020-12-03 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:30 | 2020-12-14 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:29 | 2020-12-25 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:28 | 2021-01-27 | Descending | HH/HV | 37.8° | 2.9 m | 6.6 m |
| PAZ | 05:44:28 | 2021-02-07 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:33:57 | 2021-02-14 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:27 | 2021-02-18 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:33:57 | 2021-02-25 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:27 | 2021-03-01 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:33:57 | 2021-03-08 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:27 | 2021-03-12 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:33:57 | 2021-03-19 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:28 | 2021-03-23 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:33:58 | 2021-03-30 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:29 | 2021-04-03 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 05:44:29 | 2021-04-14 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 05:44:30 | 2021-04-25 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:00 | 2021-05-02 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:30 | 2021-05-06 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:00 | 2021-05-13 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:30 | 2021-05-17 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:01 | 2021-05-24 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:31 | 2021-05-28 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:01 | 2021-06-04 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:32 | 2021-06-08 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:01 | 2021-06-15 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:32 | 2021-06-19 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:02 | 2021-06-26 | Ascending | HH | 54.0° | 2.2 m | 3.3 m |
| PAZ | 05:44:32 | 2021-06-30 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 05:44:32 | 2021-07-11 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 05:44:32 | 2021-07-22 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:02 | 2021-07-29 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:32 | 2021-08-02 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:34:03 | 2021-08-09 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:33 | 2021-08-13 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:34:03 | 2021-08-20 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 17:34:04 | 2021-08-31 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:35 | 2021-09-04 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:34:05 | 2021-09-11 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:35 | 2021-09-15 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:34:06 | 2021-09-22 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:35 | 2021-09-26 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:34:05 | 2021-10-03 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:36 | 2021-10-07 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:34:06 | 2021-10-14 | Ascending | HH | 53.9° | 2.2 m | 3.3 m |
| PAZ | 05:44:37 | 2021-10-18 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 05:44:36 | 2021-10-29 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:25:32 | 2021-10-31 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:36 | 2021-11-09 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 05:44:36 | 2021-11-20 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:32 | 2021-11-22 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 17:25:32 | 2021-11-22 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:30 | 2022-02-27 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:25 | 2022-03-01 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:34 | 2022-06-06 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:30 | 2022-06-08 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:36 | 2022-08-22 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:25:33 | 2022-08-24 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:37 | 2022-09-02 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:33 | 2022-09-04 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:38 | 2022-09-13 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:33 | 2022-09-15 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:38 | 2022-09-24 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:34 | 2022-09-26 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:38 | 2022-10-05 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:25:34 | 2022-10-07 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:38 | 2022-10-16 | Descending | HH | 37.3° | 2.9 m | 3.3 m |
| PAZ | 17:25:35 | 2022-10-18 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:39 | 2022-10-27 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:34 | 2022-10-29 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |
| PAZ | 05:44:36 | 2022-12-21 | Descending | HH | 37.4° | 2.9 m | 3.3 m |
| PAZ | 17:25:33 | 2022-12-23 | Ascending | HH | 44.6° | 2.5 m | 3.3 m |

