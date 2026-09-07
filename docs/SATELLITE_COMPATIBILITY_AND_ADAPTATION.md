# Satellite Image Compatibility & Adaptation Layer

SatQuery-AI provides a deterministic, scientifically grounded image validation and adaptation pipeline. It bridges heterogeneous satellite image uploads with specialized downstream remote sensing models.

> [!IMPORTANT]
> **Core Principle: Flexible Input Support $\neq$ Universal Model Accuracy**
> Supporting diverse image formats (GeoTIFF, uint16, SAR) ensures that files can be parsed and adapted safely without crashing. However, adaptation **does not** magically make an arbitrary satellite image compatible with a model trained on a different sensor or domain. SatQuery-AI enforces explicit domain guardrails and records structured limitations when imagery or queries fall outside validated specialist bounds.

---

## 1. Supported Inputs & Formats

| Format | Extensions | Primary Engine | Fallback Engine | Metadata Extraction |
| :--- | :--- | :--- | :--- | :--- |
| **PNG** | `.png` | Pillow | — | Width, height, channels, 8-bit RGB/RGBA |
| **JPEG** | `.jpg`, `.jpeg` | Pillow | — | Width, height, channels, 8-bit RGB |
| **TIFF** | `.tif`, `.tiff` | Rasterio | Pillow | Dtype, bit depth (8/16/32), band count |
| **GeoTIFF** | `.tif`, `.tiff`, `.gtiff` | Rasterio | Pillow (TIFF tags) | CRS, bounds, pixel resolution, geotransform, nodata |
| **WebP / BMP**| `.webp`, `.bmp` | Pillow | — | Standard raster formats |

---

## 2. Supported Modalities & Detection Rules

Sensor and modality identification strictly avoids fabricated metadata:
- If metadata or standardized filename patterns confirm the sensor, it is reported (e.g. `sentinel-2`, `sentinel-1`, `landsat`).
- If metadata is insufficient, the system designates `sensor: "unknown"`, `modality: "inferred/unknown"` rather than guessing.

| Inferred Modality | Criteria | Handling |
| :--- | :--- | :--- |
| **RGB Optical** | 3 channels (RGB) or 4 channels (RGBA) | Passed to optical VLM and change detector. |
| **Multispectral Optical** | $>3$ spectral bands or designated Sentinel-2 / Landsat bands | Adaptively selects natural RGB triplet (or B04, B03, B02). |
| **SAR Backscatter** | Radar metadata, filenames (`_VV`, `_VH`), or calibrated negative dB values | Preserves VV/VH channels, handles decibels vs linear power. |
| **Grayscale / Panchromatic** | 1 band | Replicated across 3 channels to form pseudo-RGB for VLMs. |
| **Unknown** | Metadata insufficient to establish modality | Evaluated on raw raster characteristics with explicit warning. |

---

## 3. Deterministic Adaptations

| Source Data Type | Target Representation | Method | Telemetry Key |
| :--- | :--- | :--- | :--- |
| **uint8 RGB [0, 255]** | `float32` in `[0.0, 1.0]` | Linear division: $x / 255.0$ | `uint8_div_255` |
| **uint16 Optical ($>255$)**| `float32` in `[0.0, 1.0]` | Scaled by $1/10000$ (Sentinel-2 L2A) or $1/65535$ | `uint16_scale_10000` |
| **SAR Linear Power ($>0$)** | `float32` in `dB` | $10 \cdot \log_{10}(I + 10^{-6})$ clipped to $[-45, 15]$ | `linear_power_to_db_10log10` |
| **SAR Already in dB ($< -1$)** | `float32` in `dB` | **Preserved directly** — never re-logged | `sar_db_preservation` |
| **Single-band Grayscale** | `float32` $(H, W, 3)$ | Replicated across channels | `selected_bands: [1, 1, 1]` |
| **Mismatched Dimensions** | Unified $(H, W)$ | Deterministic bilinear resampling of Image B to match Image A | `resampling: bilinear` |

---

## 4. Compatibility Status Taxonomy

The `SatelliteCompatibilityService` outputs five distinct states:
- `COMPATIBLE`: Imagery meets all specialist requirements directly (e.g. standard 8-bit RGB for LEVIR-CD).
- `ADAPTABLE`: Imagery requires deterministic transformations (e.g. uint16 scaling, dimension matching, SAR dB conversion).
- `UNSUPPORTED`: Input modality or domain is not supported by the requested specialist (e.g. SAR image supplied for bi-temporal optical change detection).
- `INSUFFICIENT_METADATA`: Metadata is missing or unverified, requiring careful execution with documented warnings.
- `INVALID`: Image is corrupted, empty (0 bytes), unreadable, or consists of $>95\%$ nodata/NaN values.

---

## 5. Current Specialist Model Domains & Guardrails

| Specialist | Current Validated Domain | Refusal / Guardrail Behavior |
| :--- | :--- | :--- |
| **Change Detection (Siamese U-Net)** | **LEVIR-CD optical building and structural change** | Flags and refuses SAR-only change pairs. For queries targeting flood, wildfire, agriculture, or mining, documents domain mismatch limitations rather than claiming general accuracy. |
| **Change VQA (SmolVLM + LoRA)** | **Bi-temporal optical change interpretation** (2-panel T1\|T2 reasoning) | Preserves detector telemetry outside VLM prompt. Requires optical inputs. |
| **Optical + SAR Fusion (Gated Dual-Branch)**| **Joint Sentinel-1 VV/VH + Sentinel-2 optical bands** | Requires at least one Optical raster and one SAR raster. Validated locally on authentic BigEarthNet pairs. |
| **Single-image VQA (SmolVLM + LoRA)** | **BigEarthNet land-cover / remote sensing VQA** | Replicates grayscale or selects RGB bands from multispectral imagery. |

---

## 6. Anti-Fabrication Guarantees

1. **No Fabricated Confidence**: Output confidence scores are strictly `null` unless calibrated probabilities exist.
2. **No Invented Temporal Ordering**: If acquisition dates are missing from metadata/filenames, the system reports `temporal_status: "unknown"` and does not guess which image is "before" or "after".
3. **No Silent Model Misuse**: Unsupported modalities or non-overlapping images (0% overlap) return structured notices and limitations rather than running models blindly.
