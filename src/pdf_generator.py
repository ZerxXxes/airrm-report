"""
PDF Report generation module.

This module generates professional PDF reports for AI-RRM metrics with
enhanced formatting for insights and detailed building information.
Includes full Cisco branding with logo, colors, and typography.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas

from data_collector import BuildingMetrics

logger = logging.getLogger(__name__)


# Official Cisco Brand Color Palette
COLORS = {
    # Primary Cisco brand colors
    "cisco_blue": colors.HexColor("#049fd9"),  # Cisco Blue (core brand)
    "cisco_blue_dark": colors.HexColor("#015269"),  # Cisco Blue 80
    "cisco_blue_light": colors.HexColor("#a8ecff"),  # Cisco Blue 20
    "med_blue": colors.HexColor("#0a60ff"),  # Medium Blue 50
    "med_blue_dark": colors.HexColor("#0e3a99"),  # Medium Blue 70
    "midnight_blue": colors.HexColor("#07182d"),  # Midnight Blue
    # Status colors
    "success": colors.HexColor("#2c6611"),  # Green 60
    "success_light": colors.HexColor("#c9f2b6"),  # Green 10
    "warning": colors.HexColor("#ff9000"),  # Orange 50
    "warning_light": colors.HexColor("#ffedd6"),  # Orange 10
    "danger": colors.HexColor("#d91821"),  # Red 50
    "danger_light": colors.HexColor("#ffe8ea"),  # Red 05
    # Grays and neutrals
    "gray_dark": colors.HexColor("#26384f"),  # Gray 90
    "gray_medium": colors.HexColor("#667180"),  # Gray 50
    "gray_light": colors.HexColor("#e1e6eb"),  # Gray 10
    "gray_lighter": colors.HexColor("#f2f5f7"),  # Gray 05
    "border": colors.HexColor("#b9c1c9"),  # Gray 20
    # Text colors
    "text_dark": colors.HexColor("#26384f"),  # Gray 90
    "text_medium": colors.HexColor("#536070"),  # Gray 60
    "text_light": colors.HexColor("#848d99"),  # Gray 40
    # Special
    "white": colors.white,
    "black": colors.black,
}


# Building-wide insight types that should only be shown once per building
# These are not specific to any frequency band
#
# INSIGHT DEDUPLICATION STRATEGY:
# -------------------------------
# AI-RRM insights fall into two categories:
#
# 1. BUILDING-WIDE INSIGHTS (show once per building):
#    - Apply to entire building configuration, not individual frequency bands
#    - Examples: 'busy-hours' (RRM timing configuration is building-level)
#    - These are deduplicated by insight type to avoid repetition
#
# 2. BAND-SPECIFIC INSIGHTS (show per frequency):
#    - Specific to individual frequency bands (2.4, 5, 6 GHz)
#    - Examples: channel interference, DFS issues, utilization warnings
#    - These are grouped under their respective frequency band sections
#
# Implementation: See _add_issues_section() method which separates and
# renders these insight types appropriately in the PDF report.
BUILDING_WIDE_INSIGHTS = {
    "busy-hours",  # RRM busy hours configuration applies to entire building
    # Add other building-wide insight types as they are identified
}


class PDFReportGenerator:
    """
    Generates professional PDF reports for AI-RRM metrics.

    Creates multi-section reports with executive summary, building
    details, and comprehensive insights with improved formatting.
    """

    def __init__(self, output_path: str, logo_path: Optional[str] = None) -> None:
        """
        Initialize PDF report generator with Cisco branding.

        Parameters:
            output_path (str): Path where PDF will be saved
            logo_path (Optional[str]): Path to Cisco logo image file (PNG/JPG recommended)
                                      If None, report is generated without logo

        Returns:
            None
        """
        self.output_path: str = output_path
        self.logo_path: Optional[str] = logo_path

        # Validate logo path if provided
        if self.logo_path and not os.path.exists(self.logo_path):
            logger.warning(
                f"Logo file not found: {self.logo_path}. Generating report without logo."
            )
            self.logo_path = None

        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=1.0 * inch,  # Increased for branded header
            bottomMargin=1.0 * inch,  # Increased for branded footer
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )
        self.styles = getSampleStyleSheet()
        self.story: List[Any] = []
        self.page_width = letter[0] - 1.5 * inch  # usable width

        # For PDF bookmarks
        self.bookmarks: List[tuple] = []  # (title, level, key)

        # Enhanced custom styles
        self.title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=28,
            textColor=COLORS["cisco_blue"],
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            leading=34,
        )

        self.subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=self.styles["Normal"],
            fontSize=11,
            textColor=COLORS["text_medium"],
            alignment=TA_CENTER,
            spaceAfter=30,
        )

        self.heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading2"],
            fontSize=16,
            textColor=COLORS["cisco_blue_dark"],
            spaceAfter=12,
            spaceBefore=20,
            fontName="Helvetica-Bold",
            borderWidth=0,
            borderPadding=8,
            borderColor=COLORS["cisco_blue"],
            leftIndent=0,
            borderRadius=2,
        )

        self.subheading_style = ParagraphStyle(
            "CustomSubHeading",
            parent=self.styles["Heading3"],
            fontSize=12,
            textColor=COLORS["gray_dark"],
            spaceAfter=8,
            spaceBefore=12,
            fontName="Helvetica-Bold",
        )

        self.normal_style = ParagraphStyle(
            "CustomNormal",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=COLORS["text_dark"],
            alignment=TA_LEFT,
            spaceAfter=6,
        )

        self.intro_style = ParagraphStyle(
            "IntroText",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=COLORS["text_dark"],
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14,
        )

        self.insight_style = ParagraphStyle(
            "InsightText",
            parent=self.styles["Normal"],
            fontSize=9,
            leftIndent=15,
            rightIndent=15,
            spaceAfter=8,
            spaceBefore=4,
            textColor=COLORS["text_dark"],
            leading=11,
        )

        self.footer_style = ParagraphStyle(
            "Footer",
            parent=self.styles["Normal"],
            fontSize=8,
            textColor=COLORS["border"],
            alignment=TA_CENTER,
        )

    @staticmethod
    def get_health_score_color(score: float) -> tuple:
        """
        Get color and status for health score.

        Parameters:
            score (float): Health score (0-100)

        Returns:
            tuple: (background_color, text_color, status_icon, status_text)
        """
        if score >= 90:
            return (COLORS["success_light"], COLORS["success"], "✓", "Excellent")
        elif score >= 80:
            return (COLORS["success_light"], COLORS["text_dark"], "✓", "Good")
        elif score >= 60:
            return (COLORS["warning_light"], COLORS["text_dark"], "⚠", "Fair")
        else:
            return (COLORS["danger_light"], COLORS["danger"], "✗", "Poor")

    def _add_bookmark(self, title: str, level: int = 0) -> Paragraph:
        """
        Create a paragraph with bookmark for PDF navigation.

        Parameters:
            title (str): Bookmark/heading title
            level (int): Heading level (0=main, 1=sub)

        Returns:
            Paragraph: Styled paragraph with bookmark
        """
        style = self.heading_style if level == 0 else self.subheading_style
        # Create paragraph with bookmark
        para = Paragraph(f'<a name="{title}"/>{title}', style)
        return para

    def generate_report(
        self,
        metrics: List[BuildingMetrics],
        summary_stats: Dict[str, Any],
        configured_bands: str = "2.4,5,6",
    ) -> None:
        """
        Generate complete PDF report with enhanced insights formatting.

        Parameters:
            metrics (List[BuildingMetrics]): List of collected metrics
            summary_stats (Dict[str, Any]): Summary statistics
            configured_bands (str): Raw FREQUENCY_BANDS env value for
                display in the report header (e.g. "2.4,5,6" or "5,6")

        Returns:
            None
        """
        logger.info(f"Generating PDF report: {self.output_path}")

        # Title page with branding
        self._add_title_page(summary_stats, configured_bands)

        # Executive Summary with KPIs and charts
        self._add_executive_summary(summary_stats, metrics)

        # Buildings with Issues (with inline insights)
        buildings_with_issues = [m for m in metrics if m.has_issues]
        if buildings_with_issues:
            self._add_issues_section(buildings_with_issues)
        else:
            self._add_no_issues_section()

        # All Buildings Summary Table
        self._add_all_buildings_table(metrics)

        # Build PDF with branded header/footer
        self.doc.build(
            self.story,
            onFirstPage=self._add_page_branding,
            onLaterPages=self._add_page_branding,
        )
        logger.info(f"Report generated successfully: {self.output_path}")

    def _add_page_branding(self, canvas_obj, doc) -> None:
        """
        Add Cisco branded header and footer to each page.

        Parameters:
            canvas_obj: ReportLab canvas object
            doc: Document template

        Returns:
            None
        """
        canvas_obj.saveState()

        # Page dimensions
        page_width, page_height = letter

        # Branded header line (Cisco blue)
        canvas_obj.setStrokeColor(COLORS["cisco_blue"])
        canvas_obj.setLineWidth(3)
        canvas_obj.line(
            0.5 * inch,
            page_height - 0.5 * inch,
            page_width - 0.5 * inch,
            page_height - 0.5 * inch,
        )

        # Footer with Cisco branding
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(COLORS["gray_medium"])

        # Cisco brand text in footer
        canvas_obj.setFillColor(COLORS["cisco_blue"])
        canvas_obj.setFont("Helvetica-Bold", 9)
        canvas_obj.drawString(0.75 * inch, 0.5 * inch, "Cisco")

        # Separator
        canvas_obj.setFillColor(COLORS["gray_medium"])
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(1.1 * inch, 0.5 * inch, "|")

        # Report title
        canvas_obj.drawString(1.3 * inch, 0.5 * inch, "AI-RRM Performance Report")

        # Page number (right-aligned)
        page_num = f"Page {doc.page}"
        canvas_obj.drawRightString(page_width - 0.75 * inch, 0.5 * inch, page_num)

        # Footer line (Cisco blue)
        canvas_obj.setStrokeColor(COLORS["cisco_blue"])
        canvas_obj.setLineWidth(1)
        canvas_obj.line(0.5 * inch, 0.65 * inch, page_width - 0.5 * inch, 0.65 * inch)

        canvas_obj.restoreState()

    def _create_stat_box(self, label: str, value: str, color: colors.Color) -> Table:
        """Create a colored stat box for KPI display."""
        data = [[value], [label]]
        table = Table(data, colWidths=[1.8 * inch], rowHeights=[0.5 * inch, 0.3 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), color),
                    ("BACKGROUND", (0, 1), (0, 1), COLORS["white"]),
                    ("TEXTCOLOR", (0, 0), (0, 0), COLORS["white"]),
                    ("TEXTCOLOR", (0, 1), (0, 1), COLORS["text_dark"]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 24),
                    ("FONTNAME", (0, 1), (0, 1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (0, 1), 9),
                    ("BOX", (0, 0), (-1, -1), 2, color),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return table

    def _add_title_page(
        self, stats: Dict[str, Any], configured_bands: str = "2.4,5,6"
    ) -> None:
        """Add professional title page with Cisco branding and logo."""
        # Cisco Logo (if available)
        if self.logo_path:
            try:
                logo = Image(
                    self.logo_path,
                    width=2.5 * inch,
                    height=0.8 * inch,
                    kind="proportional",
                )
                logo.hAlign = "CENTER"
                self.story.append(Spacer(1, 0.3 * inch))
                self.story.append(logo)
                self.story.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                logger.warning(f"Failed to load logo image: {e}")
                self.story.append(Spacer(1, 0.5 * inch))
        else:
            # Cisco text branding if no logo
            cisco_brand = Paragraph(
                "<font color='#049fd9' size='24'><b>Cisco</b></font>",
                self.subtitle_style,
            )
            self.story.append(Spacer(1, 0.3 * inch))
            self.story.append(cisco_brand)
            self.story.append(Spacer(1, 0.2 * inch))

        # Main title
        title = Paragraph("AI-RRM Performance Report", self.title_style)
        self.story.append(title)

        # Subtitle with date
        timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
        subtitle = Paragraph(
            f"Network Intelligence Report<br/>Generated on {timestamp}",
            self.subtitle_style,
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.3 * inch))

        # Decorative line using Cisco brand colors
        line_data = [["  "]]
        line = Table(line_data, colWidths=[self.page_width])
        line.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 3, COLORS["cisco_blue"]),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, COLORS["med_blue"]),
                ]
            )
        )
        self.story.append(line)
        self.story.append(Spacer(1, 0.4 * inch))

        # Introduction Section
        intro_heading = Paragraph("<b>About This Report</b>", self.subheading_style)
        self.story.append(intro_heading)
        self.story.append(Spacer(1, 0.1 * inch))

        intro_para = Paragraph(
            "This AI-Enhanced Radio Resource Management (AI-RRM) Performance Report provides "
            "comprehensive insights into the wireless network health and optimization status "
            "across your deployment. This automated report analyzes key performance indicators "
            "from Cisco Catalyst Center to help network administrators identify, prioritize, "
            "and resolve wireless infrastructure issues.",
            self.intro_style,
        )

        band_display_map = {
            "2.4": "2.4 GHz",
            "5": "5 GHz",
            "5.0": "5 GHz",
            "6": "6 GHz",
            "6.0": "6 GHz",
        }
        band_labels = [
            band_display_map.get(b.strip(), b.strip())
            for b in configured_bands.split(",")
            if b.strip()
        ]
        bands_display = ", ".join(band_labels) if band_labels else configured_bands

        config_para = Paragraph(
            f"<b>Configured Frequency Bands:</b> {bands_display}",
            self.intro_style,
        )

        intro_table = Table([[intro_para], [config_para]], colWidths=[self.page_width])
        intro_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["gray_lighter"]),
                    ("BOX", (0, 0), (-1, -1), 1, COLORS["border"]),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, COLORS["border"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        self.story.append(intro_table)
        self.story.append(Spacer(1, 0.5 * inch))

    def _add_executive_summary(
        self, stats: Dict[str, Any], metrics: List[BuildingMetrics]
    ) -> None:
        """Add executive summary section with KPI boxes and key metrics."""
        self.story.append(self._add_bookmark("Executive Summary", level=0))
        self.story.append(Spacer(1, 0.15 * inch))

        # Key Performance Indicators in colored boxes
        kpi_row1 = [
            self._create_stat_box(
                "Total Buildings",
                str(stats.get("total_buildings", 0)),
                COLORS["cisco_blue"],
            ),
            Spacer(0.2 * inch, 0),
            self._create_stat_box(
                "Buildings with Insights",
                str(stats.get("buildings_with_issues", 0)),
                COLORS["danger"]
                if stats.get("buildings_with_issues", 0) > 0
                else COLORS["success"],
            ),
            Spacer(0.2 * inch, 0),
            self._create_stat_box(
                "Total Insights",
                str(stats.get("total_insights", 0)),
                COLORS["warning"]
                if stats.get("total_insights", 0) > 0
                else COLORS["success"],
            ),
        ]

        kpi_table1 = Table(
            [kpi_row1],
            colWidths=[1.8 * inch, 0.2 * inch, 1.8 * inch, 0.2 * inch, 1.8 * inch],
        )
        kpi_table1.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        self.story.append(kpi_table1)
        self.story.append(Spacer(1, 0.2 * inch))

        kpi_row2 = [
            self._create_stat_box(
                "Total Access Points",
                str(stats.get("total_aps", 0)),
                COLORS["med_blue"],
            ),
            Spacer(0.2 * inch, 0),
            self._create_stat_box(
                "Total Clients", str(stats.get("total_clients", 0)), COLORS["med_blue"]
            ),
            Spacer(0.2 * inch, 0),
            self._create_stat_box(
                "Avg Health Score",
                f"{stats.get('average_health_score', 0):.1f}",
                self._get_health_color(stats.get("average_health_score", 0)),
            ),
        ]

        kpi_table2 = Table(
            [kpi_row2],
            colWidths=[1.8 * inch, 0.2 * inch, 1.8 * inch, 0.2 * inch, 1.8 * inch],
        )
        kpi_table2.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        self.story.append(kpi_table2)
        self.story.append(Spacer(1, 0.3 * inch))

    def _get_health_color(self, score: float) -> colors.Color:
        """Return color based on health score."""
        if score >= 80:
            return COLORS["success"]
        elif score >= 60:
            return COLORS["warning"]
        else:
            return COLORS["danger"]

    def _add_no_issues_section(self) -> None:
        """Add section when no issues are found."""
        heading = Paragraph("Buildings Requiring Attention", self.heading_style)

        success_msg = Paragraph(
            "✓ <b>All buildings are operating within normal parameters.</b><br/>"
            "No critical issues or insights requiring immediate attention were detected.",
            self.normal_style,
        )

        # Success box
        success_data = [[success_msg]]
        success_table = Table(success_data, colWidths=[self.page_width])
        success_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["gray_light"]),
                    ("BOX", (0, 0), (-1, -1), 2, COLORS["success"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                ]
            )
        )

        # Keep heading and message together
        content = KeepTogether([heading, success_table, Spacer(1, 0.3 * inch)])
        self.story.append(content)

    def _add_issues_section(self, buildings_with_issues: List[BuildingMetrics]) -> None:
        """
        Add section for buildings requiring attention with insights.

        Shows each building with its metrics and lists all insights
        inline for immediate visibility.

        Parameters:
            buildings_with_issues (List[BuildingMetrics]): Buildings
                flagged as having issues

        Returns:
            None
        """
        section_heading = self._add_bookmark("Buildings Requiring Attention", level=0)

        # Group by building
        building_dict: Dict[str, List[BuildingMetrics]] = {}
        for metric in buildings_with_issues:
            if metric.building_name not in building_dict:
                building_dict[metric.building_name] = []
            building_dict[metric.building_name].append(metric)

        for idx, (building_name, building_metrics) in enumerate(
            building_dict.items(), 1
        ):
            # Create a group to keep building info together
            building_content = []

            # Add section heading before first building only
            if idx == 1:
                building_content.append(section_heading)
                building_content.append(Spacer(1, 0.15 * inch))

            # Building header with number badge and bookmark
            building_header = Paragraph(
                f'<a name="building_{idx}"/><b>{idx}. {building_name}</b>',
                self.subheading_style,
            )
            building_content.append(building_header)

            # Hierarchy info with icon
            hierarchy = building_metrics[0].building_hierarchy
            hierarchy_para = Paragraph(f"📍 <i>{hierarchy}</i>", self.normal_style)
            building_content.append(hierarchy_para)
            building_content.append(Spacer(1, 0.1 * inch))

            # Metrics table for this building with enhanced styling
            data = [
                [
                    "Frequency Band",
                    "Health Score",
                    "RRM Changes",
                    "Access Points",
                    "Connected Clients",
                ]
            ]

            for m in sorted(building_metrics, key=lambda x: x.frequency_band):
                bg_color, text_color, icon, status = self.get_health_score_color(
                    m.rrm_health_score
                )

                data.append(
                    [
                        m.frequency_label,
                        f"{icon} {m.rrm_health_score:.1f}",
                        str(m.rrm_changes),
                        str(m.ap_count),
                        str(m.client_count),
                    ]
                )

            table = Table(
                data,
                colWidths=[1.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch],
            )

            # Build row-specific backgrounds based on health scores
            row_styles = [
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["med_blue"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, COLORS["border"]),
                ("LINEBELOW", (0, 0), (-1, 0), 2, COLORS["med_blue"]),
            ]

            # Color code rows based on health scores
            for row_idx, m in enumerate(
                sorted(building_metrics, key=lambda x: x.frequency_band), 1
            ):
                bg_color, text_color, icon, status = self.get_health_score_color(
                    m.rrm_health_score
                )
                row_styles.append(("BACKGROUND", (1, row_idx), (1, row_idx), bg_color))
                row_styles.append(("TEXTCOLOR", (1, row_idx), (1, row_idx), text_color))

            table.setStyle(TableStyle(row_styles))
            building_content.append(table)
            building_content.append(Spacer(1, 0.15 * inch))

            # ═══════════════════════════════════════════════════════════
            # INSIGHT DEDUPLICATION AND CATEGORIZATION
            # ═══════════════════════════════════════════════════════════
            # This section implements the core deduplication logic for insights.
            #
            # PROBLEM: API returns same building-wide insight (e.g., busy-hours)
            # for each frequency band, resulting in duplicate entries.
            #
            # SOLUTION: Separate insights into two categories:
            # 1. Building-wide: Show once per building (deduplicated by type)
            # 2. Band-specific: Show once per frequency band
            #
            # ALGORITHM:
            # - Iterate through all frequency band metrics for this building
            # - Check each insight's type against BUILDING_WIDE_INSIGHTS set
            # - If building-wide: Store in dict keyed by type (auto-deduplicates)
            # - If band-specific: Store in dict keyed by band (preserves all)
            # - Render building-wide section first, then band-specific sections
            # ═══════════════════════════════════════════════════════════

            has_insights = any(len(m.insights) > 0 for m in building_metrics)

            if has_insights:
                # Separate building-wide from band-specific insights
                building_wide_insights = {}  # Dict to deduplicate by insight type
                band_specific_insights = {}  # Dict by band: [insights]

                for m in building_metrics:
                    band = m.frequency_band
                    if band not in band_specific_insights:
                        band_specific_insights[band] = []

                    for insight in m.insights:
                        insight_type = insight.get("insightType", "")

                        if insight_type in BUILDING_WIDE_INSIGHTS:
                            # Deduplicate building-wide insights by type
                            # Only keep first occurrence (all are identical)
                            if insight_type not in building_wide_insights:
                                building_wide_insights[insight_type] = insight
                        else:
                            # Keep all band-specific insights (not duplicates)
                            band_specific_insights[band].append(insight)

                # Display building-wide insights first (if any)
                if building_wide_insights:
                    bw_header = Paragraph(
                        "🏢 <b>Building-Wide Insights:</b>", self.normal_style
                    )
                    building_content.append(bw_header)
                    building_content.append(Spacer(1, 0.1 * inch))

                    for insight_idx, (insight_type, insight) in enumerate(
                        sorted(building_wide_insights.items()), 1
                    ):
                        insight_text = (
                            f"<b>{insight_idx}. {insight.get('insightType', 'N/A')}</b><br/>"
                            f"<b>Description:</b> {insight.get('description', 'N/A')}<br/>"
                            f"<b>Recommendation:</b> {insight.get('reason', 'N/A')}"
                        )
                        insight_para = Paragraph(insight_text, self.insight_style)

                        insight_data = [[insight_para]]
                        insight_table = Table(insight_data, colWidths=[self.page_width])
                        insight_table.setStyle(
                            TableStyle(
                                [
                                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["white"]),
                                    ("BOX", (0, 0), (-1, -1), 1, COLORS["warning"]),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 15),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                ]
                            )
                        )

                        building_content.append(insight_table)
                        building_content.append(Spacer(1, 0.08 * inch))

                    building_content.append(Spacer(1, 0.15 * inch))

                # Display band-specific insights (if any)
                has_band_specific = any(
                    len(insights) > 0 for insights in band_specific_insights.values()
                )

                if has_band_specific:
                    bs_header = Paragraph(
                        "📡 <b>Frequency-Specific Insights:</b>", self.normal_style
                    )
                    building_content.append(bs_header)
                    building_content.append(Spacer(1, 0.1 * inch))

                    # Get frequency labels for enabled bands
                    band_labels = {
                        m.frequency_band: m.frequency_label for m in building_metrics
                    }

                    for band in sorted(band_specific_insights.keys()):
                        insights = band_specific_insights[band]
                        if insights:
                            # Frequency band label with background
                            freq_label = band_labels.get(band, f"Band {band}")
                            freq_data = [
                                [Paragraph(f"<b>{freq_label}</b>", self.normal_style)]
                            ]
                            freq_table = Table(
                                freq_data, colWidths=[self.page_width - 30]
                            )
                            freq_table.setStyle(
                                TableStyle(
                                    [
                                        (
                                            "BACKGROUND",
                                            (0, 0),
                                            (-1, -1),
                                            COLORS["med_blue"],
                                        ),
                                        (
                                            "TEXTCOLOR",
                                            (0, 0),
                                            (-1, -1),
                                            COLORS["white"],
                                        ),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                    ]
                                )
                            )
                            building_content.append(freq_table)
                            building_content.append(Spacer(1, 0.05 * inch))

                            for insight_idx, insight in enumerate(insights, 1):
                                insight_text = (
                                    f"<b>{insight_idx}. {insight.get('insightType', 'N/A')}</b><br/>"
                                    f"<b>Description:</b> {insight.get('description', 'N/A')}<br/>"
                                    f"<b>Recommendation:</b> {insight.get('reason', 'N/A')}"
                                )
                                insight_para = Paragraph(
                                    insight_text, self.insight_style
                                )

                                insight_data = [[insight_para]]
                                insight_table = Table(
                                    insight_data, colWidths=[self.page_width - 30]
                                )
                                insight_table.setStyle(
                                    TableStyle(
                                        [
                                            (
                                                "BACKGROUND",
                                                (0, 0),
                                                (-1, -1),
                                                COLORS["white"],
                                            ),
                                            (
                                                "BOX",
                                                (0, 0),
                                                (-1, -1),
                                                1,
                                                COLORS["warning"],
                                            ),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 15),
                                            ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                                            ("TOPPADDING", (0, 0), (-1, -1), 10),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                        ]
                                    )
                                )

                                building_content.append(insight_table)
                                building_content.append(Spacer(1, 0.08 * inch))

                building_content.append(Spacer(1, 0.1 * inch))
            else:
                # No insights but flagged due to low health/high changes
                reason_parts = []
                if any(m.rrm_health_score < 70 for m in building_metrics):
                    reason_parts.append("low health score")
                if any(m.rrm_changes > 100 for m in building_metrics):
                    reason_parts.append("high RRM changes")

                reason_text = f"ℹ No specific insights available. Flagged due to {' and '.join(reason_parts)}."

                reason_para = Paragraph(reason_text, self.normal_style)
                reason_data = [[reason_para]]
                reason_table = Table(reason_data, colWidths=[self.page_width])
                reason_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), COLORS["gray_light"]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 15),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                            ("TOPPADDING", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )
                building_content.append(reason_table)
                building_content.append(Spacer(1, 0.15 * inch))

            # Add separator line
            if idx < len(building_dict):
                separator = Table([["  "]], colWidths=[self.page_width])
                separator.setStyle(
                    TableStyle(
                        [
                            ("LINEABOVE", (0, 0), (-1, 0), 1, COLORS["border"]),
                        ]
                    )
                )
                building_content.append(Spacer(1, 0.2 * inch))
                building_content.append(separator)
                building_content.append(Spacer(1, 0.2 * inch))

            # Keep building content together on same page when possible
            try:
                self.story.append(KeepTogether(building_content))
            except:
                # If content is too large to keep together, add it normally
                self.story.extend(building_content)

        self.story.append(Spacer(1, 0.3 * inch))

    def _add_all_buildings_table(self, metrics: List[BuildingMetrics]) -> None:
        """
        Add comprehensive table of all buildings with summary.

        Parameters:
            metrics (List[BuildingMetrics]): All collected metrics

        Returns:
            None
        """
        self.story.append(PageBreak())

        heading = self._add_bookmark("Complete Building Inventory", level=0)

        # Create table data
        data = [
            [
                "Building Name",
                "Frequency",
                "Health Score",
                "RRM Changes",
                "APs",
                "Clients",
                "Insights",
            ]
        ]

        for m in sorted(metrics, key=lambda x: (x.building_name, x.frequency_band)):
            # Format insights count with indicator
            insight_count = len(m.insights)
            insights_display = (
                str(insight_count) if insight_count == 0 else f"⚠ {insight_count}"
            )

            # Get health score color and icon
            bg_color, text_color, icon, status = self.get_health_score_color(
                m.rrm_health_score
            )

            data.append(
                [
                    m.building_name,
                    m.frequency_label,
                    f"{icon} {m.rrm_health_score:.1f}",
                    str(m.rrm_changes),
                    str(m.ap_count),
                    str(m.client_count),
                    insights_display,
                ]
            )

        table = Table(
            data,
            colWidths=[
                1.8 * inch,
                0.9 * inch,
                0.9 * inch,
                1 * inch,
                0.6 * inch,
                0.8 * inch,
                0.8 * inch,
            ],
            repeatRows=1,  # Repeat header on each page
        )

        # Build style with health score color coding
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["cisco_blue"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),  # Building name left-aligned
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),  # Everything else centered
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ("LINEBELOW", (0, 0), (-1, 0), 2, COLORS["cisco_blue"]),
        ]

        # Add alternating row colors and health score color coding
        for i, m in enumerate(
            sorted(metrics, key=lambda x: (x.building_name, x.frequency_band)), 1
        ):
            # Alternating background
            bg_color_alt = COLORS["white"] if i % 2 == 0 else COLORS["gray_lighter"]
            table_style.append(("BACKGROUND", (0, i), (-1, i), bg_color_alt))

            # Color code health score cell
            bg_color, text_color, icon, status = self.get_health_score_color(
                m.rrm_health_score
            )
            table_style.append(("BACKGROUND", (2, i), (2, i), bg_color))
            table_style.append(("TEXTCOLOR", (2, i), (2, i), text_color))

        table.setStyle(TableStyle(table_style))

        # Add legend with color indicators
        legend_text = (
            "<b>Legend:</b> "
            "✓ = Excellent/Good (80+) | "
            "⚠ = Fair (60-79) | "
            "✗ = Poor (<60) | "
            "⚠ in Insights column = Active insights present"
        )
        legend = Paragraph(legend_text, self.normal_style)

        legend_data = [[legend]]
        legend_table = Table(legend_data, colWidths=[self.page_width])
        legend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["gray_light"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 15),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )

        # Keep heading with at least the table header together
        content = KeepTogether([heading, Spacer(1, 0.15 * inch), table])

        self.story.append(content)
        self.story.append(Spacer(1, 0.15 * inch))
        self.story.append(legend_table)
        self.story.append(Spacer(1, 0.3 * inch))
