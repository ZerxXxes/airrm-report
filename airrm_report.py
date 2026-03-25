#!/usr/bin/env python3
"""
AI-RRM Report Generator.

Main script to collect AI-RRM metrics from Cisco DNA Center and
generate PDF performance reports. This tool automates the collection
of metrics across multiple buildings and frequency bands, highlighting
buildings that require attention.

Usage:
    python airrm_report.py [options]

Examples:
    # Basic usage with default settings
    python airrm_report.py

    # Custom output path
    python airrm_report.py -o /path/to/report.pdf

    # Debug mode
    python airrm_report.py --log-level DEBUG
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Add src directory to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from auth import DNACenterAuth
from api_client import DNACenterClient
from data_collector import DataCollector
from pdf_generator import PDFReportGenerator


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging.

    Sets up logging to both console and file with timestamps and
    severity levels. Log file is created in the current directory.

    Parameters:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
            Default: 'INFO'

    Returns:
        None
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("airrm_report.log")],
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments

    Raises:
        SystemExit: If invalid arguments provided
    """
    parser = argparse.ArgumentParser(
        description=("Generate AI-RRM performance reports from Cisco DNA Center")
    )

    parser.add_argument(
        "-o",
        "--output",
        default=(f"output/airrm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"),
        help=("Output PDF file path (default: output/airrm_report_TIMESTAMP.pdf)"),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification",
    )

    parser.add_argument(
        "--logo",
        default=None,
        help=(
            "Path to Cisco logo image file (PNG/JPG recommended). "
            "If not specified, uses LOGO_PATH environment variable or generates report without logo."
        ),
    )

    return parser.parse_args()


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Reads DNA Center connection details from environment variables
    or .env file. Validates that required variables are present.

    Required environment variables:
        - DNA_CENTER_URL: Base URL of DNA Center
        - DNA_CENTER_USERNAME: Username for authentication
        - DNA_CENTER_PASSWORD: Password for authentication

    Optional environment variables:
        - VERIFY_SSL: Whether to verify SSL certificates
            (default: false)
        - FREQUENCY_BANDS: Comma-separated list of bands to collect
            (e.g., "2.4,5,6" or "5,6"). Default: all bands
        - LOGO_PATH: Path to Cisco logo image file for PDF branding
            (PNG/JPG recommended)

    Returns:
        Dict[str, Any]: Configuration dictionary with keys:
            url, username, password, verify_ssl, enabled_bands, logo_path

    Raises:
        SystemExit: If required environment variables are missing
    """
    # Load .env file if it exists
    load_dotenv()

    config = {
        "url": os.getenv("DNA_CENTER_URL"),
        "username": os.getenv("DNA_CENTER_USERNAME"),
        "password": os.getenv("DNA_CENTER_PASSWORD"),
        "verify_ssl": os.getenv("VERIFY_SSL", "false").lower() == "true",
        "logo_path": os.getenv("LOGO_PATH"),
    }

    # Parse frequency bands configuration
    bands_str = os.getenv("FREQUENCY_BANDS", "2.4,5,6")
    enabled_bands = []
    try:
        for band in bands_str.split(","):
            band = band.strip()
            if band == "2.4":
                enabled_bands.append(1)
            elif band in ["5", "5.0"]:
                enabled_bands.append(2)
            elif band in ["6", "6.0"]:
                enabled_bands.append(3)
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Invalid frequency band '{band}' ignored")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error parsing FREQUENCY_BANDS: {e}. Using all bands.")
        enabled_bands = [1, 2, 3]

    if not enabled_bands:
        enabled_bands = [1, 2, 3]

    config["enabled_bands"] = enabled_bands

    # Validate required configuration
    missing = [
        k
        for k, v in config.items()
        if v is None and k not in ["verify_ssl", "enabled_bands", "logo_path"]
    ]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nPlease set the following environment variables:")
        print("  DNA_CENTER_URL - DNA Center URL (e.g., https://172.31.96.6)")
        print("  DNA_CENTER_USERNAME - Username for authentication")
        print("  DNA_CENTER_PASSWORD - Password for authentication")
        print("\nOr create a .env file based on .env.example")
        sys.exit(1)

    return config


def main() -> None:
    """
    Main execution function.

    Orchestrates the entire report generation process:
    1. Parse command-line arguments
    2. Load configuration from environment
    3. Authenticate with DNA Center
    4. Collect metrics from all AI-RRM buildings
    5. Generate PDF report with findings

    Returns:
        None

    Raises:
        SystemExit: On authentication failure, user cancellation,
            or unexpected errors
    """
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=== AI-RRM Report Generator ===")

    # Load configuration from environment variables
    config = load_config()

    # Override SSL verification if specified via command line
    if args.no_verify_ssl:
        config["verify_ssl"] = False

    try:
        # Step 1: Authenticate with DNA Center
        logger.info("Authenticating with DNA Center...")
        auth = DNACenterAuth(
            base_url=config["url"],
            username=config["username"],
            password=config["password"],
            verify_ssl=config["verify_ssl"],
        )

        if not auth.login():
            logger.error("Authentication failed")
            sys.exit(1)

        # Step 2: Create API client
        client = DNACenterClient(auth)

        # Step 3: Collect data from all buildings
        logger.info("Collecting AI-RRM metrics...")
        logger.info(
            f"Enabled frequency bands: "
            f"{', '.join([str(b) for b in config['enabled_bands']])}"
        )
        collector = DataCollector(client, enabled_bands=config["enabled_bands"])
        metrics = collector.collect_all_metrics()

        # Edge case: No metrics were collected
        if not metrics:
            logger.warning("No metrics collected. Exiting.")
            sys.exit(0)

        # Step 4: Calculate summary statistics
        summary_stats = collector.get_summary_stats()
        logger.info(f"Collected data for {summary_stats['total_buildings']} buildings")
        logger.info(
            f"Found {summary_stats['buildings_with_issues']} buildings with issues"
        )

        # Step 5: Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 6: Generate PDF report with Cisco branding
        logger.info("Generating PDF report...")

        # Determine logo path (CLI arg takes precedence over env var)
        logo_path = args.logo or config.get("logo_path")
        if logo_path:
            logger.info(f"Using Cisco logo: {logo_path}")
        else:
            logger.info("Generating report without logo (none specified)")

        generator = PDFReportGenerator(str(output_path), logo_path=logo_path)
        generator.generate_report(metrics, summary_stats)

        # Success! Report the results
        logger.info(f"✓ Report generated successfully: {output_path}")
        logger.info(
            f"✓ Summary: {summary_stats['total_buildings']} buildings, "
            f"{summary_stats['buildings_with_issues']} with issues, "
            f"{summary_stats['total_insights']} insights"
        )

    except KeyboardInterrupt:
        # User pressed Ctrl+C
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        # Unexpected error occurred
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
