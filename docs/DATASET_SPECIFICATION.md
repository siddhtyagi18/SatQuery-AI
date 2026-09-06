# SatQuery-AI Dataset & Data Pipeline Specifications

This document defines the physical directory layouts, raster formatting standards, radiometric calibration conventions, and validation procedures required for SatQuery-AI machine learning capabilities.

---

## 1. BigEarthNet Multimodal Dataset Pipeline

### 1.1 Dataset Architecture
BigEarthNet multimodal pairs text-based Remote Sensing Visual Question Answering (RSVQA) with 10-meter resolution Sentinel-2 multi-spectral image patches.

```
BigEarthNet/
├── BigEarthNet.txt.parquet          # QA metadata and question-answer pairs (~9.55M rows)
└── patches/                          # Directory containing Sentinel-2 patch folders
    ├── S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/
    │   ├── B02.tif                  # Blue band (10m, 120x120 pixels, uint16)
    │   ├── B03.tif                  # Green band (10m, 120x120 pixels, uint16)
    │   ├── B04.tif                  # Red band (10m, 120x120 pixels, uint16)
    │   └── B08.tif                  # Near-Infrared band (10m, 120x120 pixels, uint16)
    └── S2B_MSIL2A_20180529T094029_N9999_R036_T34UFA_45_81/
        ├── B02.tif
        ├── B03.tif
        ├── B04.tif
        └── B08.tif
```

### 1.2 Data Specifications
- **Join Key**: `patch_id` in `BigEarthNet.txt.parquet` matches the folder name on disk exactly.
- **Required Bands for RGB VLM**:
  - `B04.tif` (Red - Central wavelength 665 nm)
  - `B03.tif` (Green - Central wavelength 560 nm)
  - `B02.tif` (Blue - Central wavelength 490 nm)
- **Optional NIR Band**: `B08.tif` (842 nm) for vegetation indices and 4-band multi-spectral fusion.
- **Radiometric Format**: Sentinel-2 Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance, stored as 16-bit unsigned integers (`uint16`) with a scale factor of 10,000 (DN 1,000 = 0.10 reflectance).
- **RGB Assembly**: Assembled via 2%–98% robust percentile contrast stretching into standard 8-bit RGB `PIL.Image` objects compatible with HuggingFace processors (`SmolVLMProcessor`, `AutoProcessor`).

### 1.3 Validation Script Usage
To validate BigEarthNet metadata and downloaded image rasters:
```bash
python backend/scripts/validate_bigearthnet_multimodal.py \
    --parquet-path C:/Users/Lenovo/Downloads/BigEarthNet.txt.parquet \
    --images-root C:/Users/Lenovo/Downloads/BigEarthNet-S2 \
    --max-samples 100 \
    --split train
```

---

## 2. Optical + SAR Multi-Sensor Pipeline

### 2.1 Dataset Architecture
Pairs genuine Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) rasters with co-registered Sentinel-2 multi-spectral optical scenes.

```
Optical_SAR_Dataset/
├── optical/
│   ├── scene_001_opt.tif            # Multi-band GeoTIFF (RGB or RGBN, 10m GSD)
│   └── scene_002_opt.tif
├── sar_vv/
│   ├── scene_001_sar_vv.tif         # Sentinel-1 VV single-look complex / GRD amplitude (10m GSD)
│   └── scene_002_sar_vv.tif
└── sar_vh/                          # (Optional dual-pol cross-channel)
    ├── scene_001_sar_vh.tif         # Sentinel-1 VH cross-polarization amplitude (10m GSD)
    └── scene_002_sar_vh.tif
```

### 2.2 SAR Physics & Radiometric Formulation
1. **Intensity Calculation**:
   $$\text{Intensity} = \text{Amplitude}^2$$
2. **Calibrated Backscatter in Decibels (dB)**:
   $$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\text{Intensity} + \epsilon)$$
3. **Cross-Polarization Ratio (Volume / Canopy Scattering)**:
   $$\text{Ratio}_{\text{VH/VV}} = \frac{\text{VH}_{\text{linear}}}{\text{VV}_{\text{linear}} + \epsilon}$$
4. **Cross-Polarization Difference**:
   $$\text{Diff}_{\text{dB}} = \sigma^0_{\text{VV, dB}} - \sigma^0_{\text{VH, dB}}$$
5. **Physical Scattering Signatures**:
   - **Double-Bounce Corner Reflection** ($\sigma^0_{\text{VV}} > -5.5\text{ dB}$): High returns indicating vertical urban structures, bridges, and metallic ships.
   - **Specular Surface Reflection** ($\sigma^0_{\text{VV}} < -14.0\text{ dB}$): Low returns indicating calm water bodies and smooth runways.
   - **Volume Scattering** ($\text{Ratio}_{\text{VH/VV}} > 0.20$): Multi-bounce depolarized returns from forest canopy and biomass.

### 2.3 Validation Script Usage
To validate single pairs or batch folders of Optical + SAR data:
```bash
# Validate batch directory
python backend/scripts/validate_optical_sar_pairs.py \
    --optical-dir /data/optical \
    --sar-dir /data/sar_vv

# Validate single GeoTIFF pair with dual-polarization
python backend/scripts/validate_optical_sar_pairs.py \
    --optical-file /data/optical/scene_001.tif \
    --sar-vv-file /data/sar/scene_001_vv.tif \
    --sar-vh-file /data/sar/scene_001_vh.tif
```

---

## 3. Data Requirements Checklist Before Training

Before starting VLM fine-tuning or cross-modal model training, ensure:

| Dataset | Current State | Required Before Training |
| :--- | :--- | :--- |
| **LEVIR-CD** (Experiment 01) | ✅ Downloaded & Validated (1000+ pairs) | **None.** Experiment 01 is trained and frozen. |
| **BigEarthNet Parquet QA** | ✅ Downloaded (9.55M rows, 466.8 MB) | **None.** Metadata ready. |
| **BigEarthNet S2 Rasters** | ⏳ Not yet downloaded on disk | Download Sentinel-2 patch archive (`B02`, `B03`, `B04`, `B08` GeoTIFFs). |
| **Sentinel-1 SAR Rasters** | ⏳ Not yet downloaded on disk | Download calibrated Sentinel-1 GRD/RTC GeoTIFFs (VV & VH bands). |

---

## 4. Provenance & Calibration Guardrails

The SatQuery-AI pipeline implements strict runtime checks to ensure data integrity:
- **Calibrated GeoTIFF Authentication**: Checks raster data type (`float32`, `uint16`) and embedded georeferencing metadata tags (ModelPixelScale, CRS).
- **Uncalibrated Proxy Warning**: When an 8-bit image (PNG/JPEG) is loaded into the SAR pipeline, it is clearly flagged as `is_calibrated_sar=False`, and telemetry values are marked as estimated.
- **Zero Fabrication Rule**: No mock values, simulated confidence scores, or fabricated tensors are injected into the reasoning pipeline.
