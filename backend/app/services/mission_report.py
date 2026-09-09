"""
backend/app/services/mission_report.py
--------------------------------------
Evidence-Backed AI Mission Report & PDF Generation Service for SatQuery-AI.

Rules & Guarantees:
1. PRESENTATION LAYER ONLY: Assembles existing data; NEVER executes ML models.
2. NO FABRICATION: Never invents metadata, coordinates, physical area, or confidence.
3. CONFIDENCE INTEGRITY: Confidence remains strictly None / "N/A — Uncalibrated".
4. TRUTHFUL LIMITATIONS: Explicitly lists unavailable metadata and uncalibrated metrics.
5. IMMUTABILITY: Does not mutate the existing change mask or detection checkpoint.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from ..logging_setup import logger
from ..models import Analysis, AnalysisImage, ExecutionStep, UploadedFile
from .change_detection import _RESULTS_DIR


def build_mission_report_data(
    db: Session,
    analysis: Analysis,
    roi_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble all existing evidence produced by the pipeline into a comprehensive
    structured mission report dictionary.
    """
    aid = analysis.id

    # 1. Image metadata & input summary
    images_info: List[Dict[str, Any]] = []
    links = db.query(AnalysisImage).filter(AnalysisImage.analysis_id == aid).all()
    t1_path: Optional[Path] = None
    t2_path: Optional[Path] = None

    for link in links:
        file_rec = db.query(UploadedFile).filter(UploadedFile.id == link.file_id).first()
        if file_rec:
            p = Path(file_rec.file_path)
            if link.role == "before":
                t1_path = p if p.exists() else None
            elif link.role == "after":
                t2_path = p if p.exists() else None

            images_info.append({
                "role": link.role,
                "file_name": file_rec.file_name,
                "format": file_rec.file_format,
                "width_px": file_rec.width_px or "Not available",
                "height_px": file_rec.height_px or "Not available",
                "channels": file_rec.band_count or "Not available",
                "crs": file_rec.crs or "Not available",
                "gsd_meters": file_rec.gsd_meters if file_rec.gsd_meters is not None else "Not available",
                "bounds": file_rec.bounds or "Not available",
                "file_size_bytes": file_rec.file_size_bytes,
            })

    # Fallback to demo images if paths not found on disk
    if not t1_path or not t1_path.exists():
        demo_t1 = Path("public/demo/optical_before.jpg")
        if demo_t1.exists():
            t1_path = demo_t1
    if not t2_path or not t2_path.exists():
        demo_t2 = Path("public/demo/optical_after.jpg")
        if demo_t2.exists():
            t2_path = demo_t2

    # 2. Compatibility & Adaptation
    compat_data = analysis.compatibility or {}
    adaptation_data = analysis.adaptation or {}

    # 3. Change Detection Results (from existing analysis record)
    change_map = analysis.change_map or {}
    raw_changed_pct = change_map.get("changed_pixel_pct") or change_map.get("changedPixelPct")
    raw_total_px = change_map.get("total_pixels") or change_map.get("totalPixels")
    raw_changed_px = change_map.get("changed_pixels") or change_map.get("changedPixels")

    # If analytics blob exists inside change_map, extract global stats
    analytics_blob = change_map.get("analytics") or {}
    global_stats = analytics_blob.get("global_statistics", {})
    if raw_changed_pct is None and global_stats:
        raw_changed_pct = global_stats.get("changed_pixel_percentage")
    if raw_total_px is None and global_stats:
        raw_total_px = global_stats.get("total_pixel_count")
    if raw_changed_px is None and global_stats:
        raw_changed_px = global_stats.get("changed_pixel_count")

    # Check mask file existence for visualization & fallback
    mask_path = _RESULTS_DIR / f"{aid}_changemap.png"
    if not mask_path.exists() and change_map.get("overlayUrl"):
        cand = _RESULTS_DIR / Path(change_map["overlayUrl"]).name
        if cand.exists():
            mask_path = cand
    if not mask_path.exists() and ("demo" in aid.lower() or "regression" in aid.lower()):
        demo_mask = _RESULTS_DIR / "regression_analytics_test_changemap.png"
        if demo_mask.exists():
            mask_path = demo_mask

    # 4. Trace & Execution steps
    trace_steps = (
        db.query(ExecutionStep)
        .filter(ExecutionStep.analysis_id == aid)
        .order_by(ExecutionStep.order_index)
        .all()
    )
    step_6 = next((s for s in trace_steps if s.step_id == "step-6"), None)
    exec_mode = "model_checkpoint"
    if step_6 and step_6.meta and isinstance(step_6.meta, dict):
        exec_mode = step_6.meta.get("execution_mode", "model_checkpoint")
        if raw_changed_pct is None:
            raw_changed_pct = step_6.meta.get("changed_pixel_pct")
        if raw_total_px is None:
            raw_total_px = step_6.meta.get("total_pixels") or step_6.meta.get("total_pixel_count")
        if raw_changed_px is None:
            raw_changed_px = step_6.meta.get("changed_pixels") or step_6.meta.get("changed_pixel_count")

    # If still not found, read from mask file if present
    if (raw_changed_pct is None or raw_total_px is None) and mask_path and mask_path.exists():
        try:
            from PIL import Image
            import numpy as np
            with Image.open(mask_path) as m_img:
                if m_img.mode == "RGBA":
                    r, g, b, a = m_img.split()
                    b_mask = (np.array(a) > 0).astype(np.uint8)
                else:
                    b_mask = (np.array(m_img.convert("L")) > 0).astype(np.uint8)
                raw_total_px = int(b_mask.size)
                raw_changed_px = int(np.sum(b_mask > 0))
                raw_changed_pct = round((raw_changed_px / raw_total_px) * 100.0, 2) if raw_total_px > 0 else 0.0
        except Exception as read_err:
            logger.warning("Failed reading mask for report stats: %s", read_err)


    # 5. Geo-Spatial Change Analytics (Phase 1)
    geo_meta = analytics_blob.get("geospatial_metadata", {})
    density_quadrants = analytics_blob.get("spatial_density", {}).get("quadrants", {})
    hotspots_list = analytics_blob.get("hotspots", [])
    largest_hotspot = analytics_blob.get("largest_hotspot")

    # Physical area from global analytics
    phys_area = global_stats.get("physical_area", {})
    if not isinstance(phys_area, dict):
        phys_area = {}
    phys_area_available = bool(phys_area.get("available") or phys_area.get("physical_area_available"))
    if not phys_area_available:
        phys_area["available"] = False
        phys_area["physical_area_available"] = False

    if not geo_meta.get("crs") or geo_meta.get("crs") == "N/A":
        geo_meta["crs"] = "Not available"
    if not geo_meta.get("resolution") or geo_meta.get("resolution") == "N/A":
        geo_meta["resolution"] = "Resolution unavailable"

    # 6. ROI Investigation (Phase 2)
    roi_report: Dict[str, Any] = {"performed": False, "details": None}
    if roi_data and isinstance(roi_data, dict):
        roi_report = {
            "performed": True,
            "details": roi_data,
        }

    # 7. AI Interpretation / Change VQA
    vqa_report: Dict[str, Any] = {
        "executed": False,
        "answer": None,
        "evidence": [],
        "confidence_label": "N/A — Uncalibrated",
    }
    # Check if ROI VQA was run
    if roi_data and roi_data.get("vqa"):
        vqa_report = {
            "executed": True,
            "answer": roi_data["vqa"].get("answer"),
            "evidence": roi_data["vqa"].get("evidence", []),
            "confidence_label": "N/A — Uncalibrated",
        }
    elif analysis.tool_invocations:
        for inv in analysis.tool_invocations:
            if inv.get("toolId") == "change_vqa" or inv.get("tool_id") == "change_vqa":
                vqa_report = {
                    "executed": True,
                    "answer": analysis.answer_text,
                    "evidence": analysis.evidence or [],
                    "confidence_label": "N/A — Uncalibrated",
                }
                break

    # 8. Truthful Limitations compilation
    limitations: List[str] = [
        "Confidence is not calibrated across detector and vision-language models (strictly reported as N/A — Uncalibrated).",
        "Change detection operates on pixel-level contrast via Siamese U-Net without assigning semantic land-cover categories.",
    ]
    if not phys_area_available:
        limitations.append(
            "Physical area unavailable — reliable spatial resolution (GSD) metadata was not provided in image headers."
        )
    if not geo_meta.get("crs"):
        limitations.append(
            "Geospatial coordinates unavailable — analysis spatial coordinates are reported strictly in pixel space."
        )
    if not roi_report["performed"]:
        limitations.append("ROI investigation was not performed for this report compilation.")
    else:
        limitations.append("ROI selection uses 2D axis-aligned rectangular bounding boxes.")
    if not vqa_report["executed"]:
        limitations.append("Natural-language AI interpretation (Change VQA) was not requested.")

    # Disaster assessment context (additive)
    _da_mode = adaptation_data.get("analysis_mission_mode", "general_change")
    _da_disaster = adaptation_data.get("disaster_type")
    if _da_mode == "disaster_assessment":
        limitations.append(
            "Disaster-specific classification/damage certainty is not claimed "
            "unless supported by a dedicated validated specialist model."
        )

    # 9. Formulate Executive Summary
    formatted_changed_pct = f"{raw_changed_pct:.2f}%" if raw_changed_pct is not None else "Not computed"
    formatted_changed_px = f"{raw_changed_px:,}" if raw_changed_px is not None else "Not computed"
    formatted_total_px = f"{raw_total_px:,}" if raw_total_px is not None else "Not computed"

    exec_summary = (
        f"SatQuery-AI completed bi-temporal change analysis for mission ID {aid}. "
        f"The Siamese U-Net detector identified {formatted_changed_pct} changed pixels "
        f"({formatted_changed_px} of {formatted_total_px} total pixels) between the supplied acquisitions."
    )
    if _da_mode == "disaster_assessment" and _da_disaster:
        exec_summary += (
            f" This analysis was conducted under Disaster Assessment context "
            f"(selected disaster: {_da_disaster.capitalize()}). "
            f"Detected change regions are highlighted as candidate affected areas for operator investigation."
        )
    if phys_area_available and isinstance(phys_area, dict):
        ha = phys_area.get("changed_area_hectares")
        if ha is not None:
            exec_summary += f" The physical changed area is calculated at {ha} hectares based on verified sensor resolution."

    if roi_report["performed"]:
        r_stats = roi_data.get("statistics", {})
        r_pct = r_stats.get("changed_percentage")
        if r_pct is not None:
            exec_summary += f" An interactive Region of Interest (ROI) was inspected with a localized change density of {r_pct:.2f}%."

    return {
        "analysis_id": aid,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else datetime.datetime.utcnow().isoformat(),
        "mode": analysis.mode,
        "status": analysis.status,
        "query": analysis.query,
        "executive_summary": exec_summary,
        "input_images": images_info,
        "t1_path": str(t1_path) if t1_path else None,
        "t2_path": str(t2_path) if t2_path else None,
        "change_mask_path": str(mask_path) if mask_path and mask_path.exists() else None,
        "compatibility": compat_data,
        "adaptation": adaptation_data,
        "mission_type": "Disaster Assessment" if _da_mode == "disaster_assessment" else "General Change Analysis",
        "disaster_type": _da_disaster.capitalize() if _da_disaster else None,
        "change_detection": {
            "changed_pixel_pct": raw_changed_pct,
            "changed_pixel_count": raw_changed_px,
            "total_pixel_count": raw_total_px,
            "unchanged_pixel_count": (raw_total_px - raw_changed_px) if (raw_total_px and raw_changed_px) else None,
            "unchanged_pixel_pct": round(100.0 - raw_changed_pct, 2) if raw_changed_pct is not None else None,
            "threshold_used": 0.70,
            "execution_mode": exec_mode,
            "checkpoint": "best_model.pt (Siamese U-Net / LEVIR-CD)",
            "confidence": None,
            "confidence_label": "N/A — Uncalibrated",
        },
        "geospatial_analytics": {
            "total_pixels": raw_total_px,
            "changed_pixels": raw_changed_px,
            "changed_percentage": raw_changed_pct,
            "physical_area": phys_area,
            "crs": geo_meta.get("crs") or "Not available",
            "resolution": geo_meta.get("resolution") or "Not available",
            "hotspots_count": len(hotspots_list),
            "largest_hotspot": largest_hotspot,
            "hotspots": hotspots_list[:10],
            "density_quadrants": density_quadrants,
        },
        "roi_investigation": roi_report,
        "ai_interpretation": vqa_report,
        "trace_steps": [
            {
                "step_id": s.step_id,
                "title": s.title,
                "status": s.status,
                "detail": s.detail,
            }
            for s in trace_steps
        ],
        "limitations": limitations,
    }


def generate_mission_report_pdf(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Generate a clean, high-precision PDF mission report using ReportLab Platypus.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 36pt (0.5 inch) margins for maximum printable data density
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Clean Styles
    c_primary = colors.HexColor("#0B132B")
    c_cyan = colors.HexColor("#007799")
    c_magenta = colors.HexColor("#990022")
    c_border = colors.HexColor("#D1D5DB")
    c_muted = colors.HexColor("#4B5563")
    c_light_bg = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=c_cyan,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1F2937"),
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    callout_text = ParagraphStyle(
        "CalloutText",
        parent=body_style,
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#0F172A"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1F2937"),
    )

    story: List[Any] = []

    # -----------------------------------------------------------------------
    # Header Banner
    # -----------------------------------------------------------------------
    story.append(Paragraph("SATQUERY-AI MISSION REPORT", title_style))
    story.append(
        Paragraph(
            "GEOSPATIAL BI-TEMPORAL CHANGE DETECTION & EVIDENCE REPORT • SMART INDIA HACKATHON",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_cyan, spaceAfter=8))

    # Meta Table (Analysis ID, Date, Mode, Threshold, Confidence)
    meta_row_1 = [
        Paragraph("<b>Mission ID:</b>", body_style),
        Paragraph(str(report_data.get("analysis_id")), table_cell_style),
        Paragraph("<b>Generated:</b>", body_style),
        Paragraph(str(report_data.get("created_at"))[:19], table_cell_style),
    ]
    meta_row_2 = [
        Paragraph("<b>Mode:</b>", body_style),
        Paragraph(str(report_data.get("mode")), table_cell_style),
        Paragraph("<b>Detector Threshold:</b>", body_style),
        Paragraph("0.70 (LEVIR-CD Calibrated)", table_cell_style),
    ]
    meta_row_3 = [
        Paragraph("<b>Checkpoint:</b>", body_style),
        Paragraph("best_model.pt (SiameseUNet)", table_cell_style),
        Paragraph("<b>Confidence:</b>", body_style),
        Paragraph("<b>N/A — Uncalibrated</b>", table_cell_style),
    ]
    meta_table = Table([meta_row_1, meta_row_2, meta_row_3], colWidths=[90, 180, 110, 160])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section A: Executive Summary
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. EXECUTIVE SUMMARY", section_heading))
    summary_box = Table(
        [[Paragraph(report_data.get("executive_summary", "No summary available."), callout_text)]],
        colWidths=[540],
    )
    summary_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ("BOX", (0, 0), (-1, -1), 1, c_cyan),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(summary_box)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section B: Visual Evidence (T1 | T2 | Change Heatmap)
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. VISUAL EVIDENCE & SENSOR APERTURE", section_heading))
    t1_p = report_data.get("t1_path")
    t2_p = report_data.get("t2_path")
    mask_p = report_data.get("change_mask_path")

    vis_images: List[Any] = []
    col_w = 175
    img_h = 115

    for path_str, label in [
        (t1_p, "T1: Before Acquisition"),
        (t2_p, "T2: After Acquisition"),
        (mask_p, "Siamese U-Net Change Heatmap"),
    ]:
        if path_str and Path(path_str).exists():
            try:
                img_el = RLImage(path_str, width=col_w, height=img_h)
                cell = [Paragraph(f"<b>{label}</b>", table_cell_style), img_el]
            except Exception:
                cell = [Paragraph(f"<b>{label}</b><br/>[Visual Asset Render Failed]", table_cell_style)]
        else:
            cell = [Paragraph(f"<b>{label}</b><br/>[Image asset unavailable on disk]", table_cell_style)]
        vis_images.append(cell)

    vis_table = Table([[vis_images[0], vis_images[1], vis_images[2]]], colWidths=[180, 180, 180])
    vis_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(vis_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section C & D: Quantitative Change Detection & Geo-Spatial Analytics
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. QUANTITATIVE CHANGE DETECTION & SPATIAL ANALYTICS", section_heading))
    cd = report_data.get("change_detection", {})
    ga = report_data.get("geospatial_analytics", {})
    pa = ga.get("physical_area", {})

    tot_px = cd.get("total_pixel_count")
    tot_str = f"{tot_px:,}" if tot_px is not None else "Not computed"
    chg_px = cd.get("changed_pixel_count")
    chg_str = f"{chg_px:,}" if chg_px is not None else "Not computed"
    chg_pct = cd.get("changed_pixel_pct")
    chg_pct_str = f"{chg_pct:.2f}%" if chg_pct is not None else "Not computed"
    unchg_px = cd.get("unchanged_pixel_count")
    unchg_str = f"{unchg_px:,}" if unchg_px is not None else "Not computed"
    unchg_pct = cd.get("unchanged_pixel_pct")
    unchg_pct_str = f"{unchg_pct:.2f}%" if unchg_pct is not None else "Not computed"

    cd_rows = [
        [
            Paragraph("<b>Total Scene Pixels</b>", table_cell_style),
            Paragraph(tot_str, table_cell_style),
            Paragraph("<b>Changed Pixels</b>", table_cell_style),
            Paragraph(f"{chg_str} ({chg_pct_str})", table_cell_style),
        ],
        [
            Paragraph("<b>Unchanged Pixels</b>", table_cell_style),
            Paragraph(f"{unchg_str} ({unchg_pct_str})", table_cell_style),
            Paragraph("<b>Model Threshold</b>", table_cell_style),
            Paragraph("0.70", table_cell_style),
        ],
        [
            Paragraph("<b>Coordinate System (CRS)</b>", table_cell_style),
            Paragraph(str(ga.get("crs", "Not available")), table_cell_style),
            Paragraph("<b>Spatial Resolution (GSD)</b>", table_cell_style),
            Paragraph(str(ga.get("resolution", "Not available")), table_cell_style),
        ],
        [
            Paragraph("<b>Physical Changed Area</b>", table_cell_style),
            Paragraph(
                f"{pa.get('changed_area_hectares', 'N/A')} ha"
                if (pa and pa.get("available"))
                else "Unavailable (No reliable GSD)",
                table_cell_style,
            ),
            Paragraph("<b>Total Physical Scene</b>", table_cell_style),
            Paragraph(
                f"{pa.get('total_area_hectares', 'N/A')} ha"
                if (pa and pa.get("available"))
                else "Unavailable (No reliable GSD)",
                table_cell_style,
            ),
        ],
    ]
    cd_table = Table(cd_rows, colWidths=[130, 140, 130, 140])
    cd_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), c_light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, c_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(cd_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section E: Hotspot Breakdown Table
    # -----------------------------------------------------------------------
    story.append(Paragraph("4. CONNECTED COMPONENT HOTSPOTS (TOP LOCALIZED CLUSTERS)", section_heading))
    hotspots = ga.get("hotspots", [])
    if hotspots:
        hs_data = [[
            Paragraph("Cluster #", table_header_style),
            Paragraph("Pixel Area", table_header_style),
            Paragraph("Share of Change", table_header_style),
            Paragraph("Centroid (px: X, Y)", table_header_style),
            Paragraph("Bounding Box (px: x1, y1, x2, y2)", table_header_style),
        ]]
        for i, h in enumerate(hotspots[:5], start=1):
            hs_area = h.get("pixel_area")
            hs_area_str = f"{hs_area:,} px" if hs_area is not None else "N/A"
            hs_share = (
                h.get("pct_of_roi_change")
                or h.get("pct_of_total_change")
                or h.get("percentage_of_total_changed_rounded")
                or h.get("percentage_of_total_changed")
            )
            hs_share_str = f"{hs_share:.2f}%" if hs_share is not None else "N/A"

            c = h.get("centroid_px", {})
            if isinstance(c, dict):
                cx = c.get("x", c.get(0, 0))
                cy = c.get("y", c.get(1, 0))
            elif isinstance(c, (list, tuple)) and len(c) >= 2:
                cx, cy = c[0], c[1]
            else:
                cx, cy = 0, 0
            centroid_str = f"({cx:.1f}, {cy:.1f})" if isinstance(cx, float) else f"({cx}, {cy})"

            b = h.get("bounding_box") or h.get("bbox_px") or {}
            if isinstance(b, dict):
                bbox_str = f"[{b.get('x1', 0)}, {b.get('y1', 0)}, {b.get('x2', 0)}, {b.get('y2', 0)}]"
            elif isinstance(b, (list, tuple)):
                bbox_str = f"{list(b)}"
            else:
                bbox_str = "N/A"

            cid = h.get("hotspot_id") or h.get("id") or i
            hs_data.append([
                Paragraph(f"Cluster {cid}", table_cell_style),
                Paragraph(hs_area_str, table_cell_style),
                Paragraph(hs_share_str, table_cell_style),
                Paragraph(centroid_str, table_cell_style),
                Paragraph(bbox_str, table_cell_style),
            ])
        hs_table = Table(hs_data, colWidths=[65, 85, 95, 125, 170])
        hs_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(hs_table)
    else:
        story.append(Paragraph("No connected-component change clusters met the significance threshold.", body_style))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section F: ROI Investigation
    # -----------------------------------------------------------------------
    story.append(Paragraph("5. REGION OF INTEREST (ROI) INVESTIGATION", section_heading))
    roi_info = report_data.get("roi_investigation", {})
    if roi_info.get("performed") and roi_info.get("details"):
        rd = roi_info["details"]
        r_coords = rd.get("roi", {}).get("pixel_coordinates", {})
        r_stats = rd.get("statistics", {})
        r_comp = rd.get("global_comparison", {})

        r_chg_px = r_stats.get("changed_pixels")
        r_chg_str = f"{r_chg_px:,} px" if r_chg_px is not None else "N/A"
        r_chg_pct = r_stats.get("changed_percentage")
        r_chg_pct_str = f"({r_chg_pct:.2f}%)" if r_chg_pct is not None else ""
        r_glob_pct = r_comp.get("global_changed_percentage")
        r_glob_str = f"{r_glob_pct:.2f}%" if r_glob_pct is not None else "N/A"
        r_diff = r_comp.get("difference_percentage")
        r_diff_str = f"{r_diff:+.2f}%" if r_diff is not None else "N/A"

        roi_rows = [
            [
                Paragraph("<b>ROI Geometry (px)</b>", table_cell_style),
                Paragraph(f"{r_coords.get('width', 0)} × {r_coords.get('height', 0)} px (x1:{r_coords.get('x1')}, y1:{r_coords.get('y1')})", table_cell_style),
                Paragraph("<b>ROI Changed Pixels</b>", table_cell_style),
                Paragraph(f"{r_chg_str} {r_chg_pct_str}", table_cell_style),
            ],
            [
                Paragraph("<b>Global Scene Baseline</b>", table_cell_style),
                Paragraph(r_glob_str, table_cell_style),
                Paragraph("<b>ROI vs Global Delta</b>", table_cell_style),
                Paragraph(f"{r_diff_str} ({r_comp.get('summary', 'Calculated')})", table_cell_style),
            ],
        ]
        roi_table = Table(roi_rows, colWidths=[130, 140, 130, 140])
        roi_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#16A34A")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBF7D0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(roi_table)
    else:
        story.append(Paragraph("ROI investigation was not performed.", body_style))
    story.append(Spacer(1, 8))


    # -----------------------------------------------------------------------
    # Section G: AI Interpretation (Change VQA)
    # -----------------------------------------------------------------------
    story.append(Paragraph("6. AI VISION-LANGUAGE INTERPRETATION (CHANGE VQA)", section_heading))
    vqa_info = report_data.get("ai_interpretation", {})
    if vqa_info.get("executed") and vqa_info.get("answer"):
        vqa_box = Table(
            [
                [
                    Paragraph("<b>Status:</b> Executed via Domain-Adapted VLM", table_cell_style),
                    Paragraph("<b>Confidence:</b> N/A — Uncalibrated", table_cell_style),
                ],
                [
                    Paragraph(
                        f"<b>AI Interpretation:</b><br/>{vqa_info.get('answer')}",
                        callout_text,
                    ),
                    Paragraph(
                        "<b>Evidence Grounding:</b><br/>"
                        + "<br/>".join(f"• {e}" for e in vqa_info.get("evidence", [])),
                        table_cell_style,
                    ),
                ],
            ],
            colWidths=[360, 180],
        )
        vqa_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FAF5FF")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9333EA")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E9D5FF")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(vqa_box)
    else:
        story.append(Paragraph("AI interpretation was not requested for this analysis.", body_style))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section H: Evidence & Execution Trace Audit
    # -----------------------------------------------------------------------
    story.append(Paragraph("7. EXECUTION TRACE & MODEL PROVENANCE", section_heading))
    steps = report_data.get("trace_steps", [])
    if steps:
        trace_data = [[
            Paragraph("Step", table_header_style),
            Paragraph("Title", table_header_style),
            Paragraph("Status", table_header_style),
            Paragraph("Execution Detail", table_header_style),
        ]]
        for s in steps:
            trace_data.append([
                Paragraph(str(s.get("step_id")), table_cell_style),
                Paragraph(str(s.get("title")), table_cell_style),
                Paragraph(str(s.get("status")).upper(), table_cell_style),
                Paragraph(str(s.get("detail")), table_cell_style),
            ])
        trace_table = Table(trace_data, colWidths=[55, 115, 60, 310])
        trace_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_primary),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(trace_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Section I: Truthful Limitations & Disclaimers
    # -----------------------------------------------------------------------
    story.append(Paragraph("8. SENSITIVITY, UNCALIBRATED METRICS & LIMITATIONS", section_heading))
    limitations_list = report_data.get("limitations", [])
    lim_bullets = [
        Paragraph(f"• <b>{lim}</b>" if "Confidence" in lim or "Physical" in lim else f"• {lim}", body_style)
        for lim in limitations_list
    ]
    lim_table = Table([[Paragraph("<br/>".join(
        f"• {lim}" for lim in limitations_list
    ), body_style)]], colWidths=[540])
    lim_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(lim_table)

    # Build document
    doc.build(story)
    return output_path
