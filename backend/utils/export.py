# utils/export.py — PDF report generation for analysis results.
# Phase 9: Post-MVP Enhancement #1.
#
# Generates a professional PDF report from the analysis pipeline output
# including summary stats, feature importance, archetypes, and campaign scores.

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)


def generate_analysis_pdf(analysis_result: dict) -> bytes:
    """Generate a PDF report from analysis results.

    Args:
        analysis_result: The full JSON response from /api/analyze.

    Returns:
        PDF file contents as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        spaceAfter=8,
        spaceBefore=16,
        textColor=colors.HexColor("#1e40af"),
    ))
    styles.add(ParagraphStyle(
        name="MetricLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        name="MetricValue",
        parent=styles["Normal"],
        fontSize=14,
        spaceAfter=4,
    ))

    elements = []

    # ── Title ─────────────────────────────────────────────────────────────
    elements.append(Paragraph("Petition Effectiveness Analysis Report", styles["Title"]))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#dbeafe")))
    elements.append(Spacer(1, 16))

    # ── Summary ───────────────────────────────────────────────────────────
    summary = analysis_result.get("summary", {})
    cv = analysis_result.get("cv_metrics", {})

    elements.append(Paragraph("Executive Summary", styles["SectionHeader"]))

    summary_data = [
        ["Campaigns Analyzed", str(summary.get("n_campaigns", "—"))],
        ["Average Conversion Rate", f"{summary.get('avg_conversion_rate', 0)}%"],
        ["Best Campaign", summary.get("best_campaign", "—")[:60]],
        ["Worst Campaign", summary.get("worst_campaign", "—")[:60]],
        ["Model Used", cv.get("model_name", "—")],
        ["Cross-Validation R²", f"{cv.get('cv_r2_mean', 0):.3f}"],
        ["Cross-Validation MAE", f"{cv.get('cv_mae_mean', 0):.3f}"],
    ]

    summary_table = Table(summary_data, colWidths=[2.5 * inch, 4 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f9ff")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # ── Feature Importance ────────────────────────────────────────────────
    features = analysis_result.get("feature_importance", [])
    if features:
        elements.append(Paragraph("Top Conversion Drivers", styles["SectionHeader"]))
        elements.append(Paragraph(
            "Features ranked by SHAP importance — these had the largest impact "
            "on predicted conversion rates in your historical data.",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 8))

        fi_header = [["Rank", "Feature", "Impact"]]
        fi_rows = [
            [str(i + 1), f.get("label", f.get("feature", "")), f"{f.get('importance_pct', 0)}%"]
            for i, f in enumerate(features[:10])
        ]

        fi_table = Table(fi_header + fi_rows, colWidths=[0.6 * inch, 4.4 * inch, 1.5 * inch])
        fi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(fi_table)
        elements.append(Spacer(1, 20))

    # ── Archetypes ────────────────────────────────────────────────────────
    archetypes = analysis_result.get("archetypes", [])
    if archetypes:
        elements.append(Paragraph("Campaign Archetypes", styles["SectionHeader"]))
        elements.append(Paragraph(
            "Your campaigns cluster into these distinct patterns based on "
            "their linguistic and structural similarities.",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 8))

        arch_header = [["Archetype", "Campaigns", "Avg Conversion", "Dominant Traits"]]
        arch_rows = [
            [
                a.get("name", ""),
                str(a.get("campaign_count", 0)),
                f"{a.get('avg_conversion_rate', 0)}%",
                ", ".join(a.get("dominant_traits", [])[:2]) or "—",
            ]
            for a in archetypes
        ]

        arch_table = Table(arch_header + arch_rows, colWidths=[1.5 * inch, 1 * inch, 1.2 * inch, 2.8 * inch])
        arch_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(arch_table)
        elements.append(Spacer(1, 20))

    # ── Campaign Leaderboard ──────────────────────────────────────────────
    campaigns = analysis_result.get("campaign_scores", [])
    if campaigns:
        elements.append(Paragraph("Campaign Leaderboard", styles["SectionHeader"]))

        camp_header = [["Campaign", "Traffic", "Actual", "Predicted", "Grade"]]
        camp_rows = [
            [
                c.get("headline", "")[:45],
                c.get("traffic_source", ""),
                f"{c.get('actual_conversion', 0)}%",
                f"{c.get('predicted_conversion', 0)}%",
                c.get("grade", ""),
            ]
            for c in campaigns[:20]
        ]

        camp_table = Table(camp_header + camp_rows, colWidths=[2.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
        camp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(camp_table)
        elements.append(Spacer(1, 20))

    # ── Source Breakdown ──────────────────────────────────────────────────
    sources = analysis_result.get("source_breakdown", [])
    if sources:
        elements.append(Paragraph("Traffic Source Breakdown", styles["SectionHeader"]))

        src_header = [["Source", "Campaigns", "Avg Conversion", "Std Dev"]]
        src_rows = [
            [
                s.get("traffic_source", ""),
                str(s.get("n_campaigns", 0)),
                f"{s.get('avg_conversion', 0)}%",
                f"±{s.get('std_conversion', 0)}%",
            ]
            for s in sources
        ]

        src_table = Table(src_header + src_rows, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 2 * inch])
        src_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(src_table)
        elements.append(Spacer(1, 20))

    # ── Footer ────────────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#dbeafe")))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<i>This report was generated by the Petition Effectiveness Analyzer. "
        "The model identifies statistical patterns — recommendations are hypotheses "
        "to test, not guaranteed rules.</i>",
        styles["Normal"],
    ))

    doc.build(elements)
    return buffer.getvalue()
