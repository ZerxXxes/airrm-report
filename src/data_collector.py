"""
Data collection module for AI-RRM metrics.

This module orchestrates the collection of AI-RRM performance metrics
from DNA Center across multiple buildings and frequency bands. It provides
data structures and collection logic for generating comprehensive reports.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from api_client import DNACenterClient

logger = logging.getLogger(__name__)


@dataclass
class BuildingMetrics:
    """
    Metrics for a single building and frequency band.

    This dataclass stores all collected metrics for one building at
    one specific frequency (2.4, 5, or 6 GHz) including performance
    data, coverage statistics, and AI-generated insights.
    """

    building_id: str
    building_name: str
    building_hierarchy: str
    profile_name: str
    frequency_band: int
    frequency_label: str

    # Coverage metrics
    ap_count: int = 0
    client_count: int = 0

    # Performance metrics
    rrm_health_score: float = 0.0
    rrm_changes: int = 0

    # Insights
    insights: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    has_issues: bool = False

    def calculate_issue_status(
        self, health_threshold: float = 70.0, changes_threshold: int = 100
    ) -> None:
        """
        Determine if building has issues based on thresholds.

        Evaluates metrics against configurable thresholds to flag
        buildings requiring attention. A building is considered to
        have issues if any of these conditions are met:
        - Health score below threshold
        - Active insights present
        - RRM changes exceed threshold

        Parameters:
            health_threshold (float): Minimum acceptable health score.
                Default: 70.0
            changes_threshold (int): Maximum acceptable RRM changes.
                Default: 100

        Returns:
            None: Sets self.has_issues flag
        """
        has_data = self.ap_count > 0 or self.rrm_health_score > 0
        self.has_issues = (
            (has_data and self.rrm_health_score < health_threshold)
            or len(self.insights) > 0
            or self.rrm_changes > changes_threshold
        )


class DataCollector:
    """
    Collects AI-RRM metrics for all enabled buildings.

    This class coordinates the entire data collection process,
    from discovering AI-RRM enabled buildings to gathering metrics
    across all frequency bands.
    """

    # Frequency band mapping (Catalyst Center API enum → display label).
    #
    # The Catalyst Center AI-RRM GraphQL API uses a sequential 1-based enum
    # for the frequencyBand parameter, NOT the literal GHz values:
    #   1 = 2.4 GHz
    #   2 = 5 GHz
    #   3 = 6 GHz
    FREQUENCY_BANDS = {1: "2.4 GHz", 2: "5 GHz", 3: "6 GHz"}

    def __init__(
        self, client: DNACenterClient, enabled_bands: Optional[List[int]] = None
    ) -> None:
        """
        Initialize data collector.

        Parameters:
            client (DNACenterClient): Authenticated DNA Center client
            enabled_bands (List[int]): List of frequency band enum values to
                collect (1=2.4 GHz, 2=5 GHz, 3=6 GHz). If None, collects all
                bands. Default: None

        Returns:
            None
        """
        self.client: DNACenterClient = client
        self.buildings: List[Dict[str, Any]] = []
        self.metrics: List[BuildingMetrics] = []

        # Configure which bands to collect
        if enabled_bands is None:
            self.enabled_bands = list(self.FREQUENCY_BANDS.keys())
        else:
            # Discard any values not in the known enum set
            self.enabled_bands = [b for b in enabled_bands if b in self.FREQUENCY_BANDS]
            if not self.enabled_bands:
                logger.warning("No valid frequency bands specified. Using all bands.")
                self.enabled_bands = list(self.FREQUENCY_BANDS.keys())

        logger.info(
            f"Data collector initialized for bands: "
            f"{', '.join([self.FREQUENCY_BANDS[b] for b in self.enabled_bands])}"
        )

    def collect_all_metrics(self) -> List[BuildingMetrics]:
        """
        Collect metrics for all AI-RRM enabled buildings.

        Orchestrates the entire collection process:
        1. Discovers buildings with AI-RRM enabled
        2. Iterates through each building
        3. Collects metrics for each frequency band (2.4, 5, 6 GHz)
        4. Handles errors gracefully, continuing on failure

        Returns:
            List[BuildingMetrics]: List of collected metrics, one entry
                per building/frequency combination

        Raises:
            None: Errors are logged but don't stop collection
        """
        logger.info("Starting data collection")

        # Perform health check before collection
        logger.info("Validating connectivity to Catalyst Center...")
        if not self.client.health_check():
            logger.error("Health check failed - aborting data collection")
            logger.error("Please verify network connectivity and credentials")
            return []

        # Get buildings with AI-RRM enabled
        try:
            self.buildings = self.client.get_airrm_buildings()
        except Exception as e:
            logger.error(f"Failed to retrieve AI-RRM buildings: {e}")
            logger.error("Cannot continue without building list")
            return []

        # Edge case: No AI-RRM buildings configured
        if not self.buildings:
            logger.warning("No AI-RRM enabled buildings found")
            logger.warning("Check that AI-RRM profiles are assigned to buildings")
            return []

        # Collect metrics for each building and frequency band
        successful_collections = 0
        failed_collections = 0

        for building in self.buildings:
            building_id: Optional[str] = building.get("instanceUUID")
            building_name = building.get("name", "Unknown")
            building_hierarchy = building.get("groupNameHierarchy", "")
            profile_name = building.get("aiRfProfileName", "Unknown")

            if not building_id:
                logger.warning(
                    f"Skipping building '{building_name}': missing instanceUUID"
                )
                continue

            # The AI-RRM profile API returns floor-level UUIDs. Resolve to
            # building level for coverage/performance GraphQL queries.
            building_uuid = self.client.resolve_building_uuid(building_id)

            logger.info(
                f"Collecting data for building: {building_name} ({building_hierarchy})"
            )

            # Iterate through enabled frequency bands only
            for freq_band, freq_label in self.FREQUENCY_BANDS.items():
                # Skip bands not enabled in configuration
                if freq_band not in self.enabled_bands:
                    logger.warning(
                        f"Skipping {freq_label} for {building_name} "
                        f"(band not enabled in configuration)"
                    )
                    continue

                try:
                    metrics = self._collect_building_frequency_metrics(
                        floor_uuid=building_id,
                        building_uuid=building_uuid,
                        building_name=building_name,
                        building_hierarchy=building_hierarchy,
                        profile_name=profile_name,
                        freq_band=freq_band,
                        freq_label=freq_label,
                    )
                    if metrics:
                        self.metrics.append(metrics)
                        successful_collections += 1
                except Exception as e:
                    # Log but continue — one failure should not stop all collection
                    failed_collections += 1
                    logger.error(
                        f"Failed to collect metrics for "
                        f"{building_name} - {freq_label}: {e}"
                    )
                    logger.debug("Error details:", exc_info=True)

        logger.info(
            f"Collection complete. Gathered metrics for "
            f"{len(self.metrics)} building/frequency combinations"
        )
        logger.info(f"Success: {successful_collections}, Failed: {failed_collections}")

        if failed_collections > 0:
            logger.warning(
                f"{failed_collections} collection(s) failed - "
                f"report will contain partial data"
            )

        return self.metrics

    def _collect_building_frequency_metrics(
        self,
        floor_uuid: str,
        building_uuid: str,
        building_name: str,
        building_hierarchy: str,
        profile_name: str,
        freq_band: int,
        freq_label: str,
    ) -> BuildingMetrics:
        """
        Collect all metrics for a single building/frequency combo.

        Makes three separate API calls to gather:
        1. Coverage data (APs, clients) — uses building-level UUID
        2. Performance data (health score, RRM changes) — uses building-level UUID
        3. Insights (AI-generated recommendations) — uses floor-level UUID

        Parameters:
            floor_uuid (str): Floor-level UUID from AI-RRM profile API
            building_uuid (str): Building-level UUID resolved via site map;
                equals floor_uuid when the site is already a building
            building_name (str): Building display name
            building_hierarchy (str): Full site hierarchy path
            profile_name (str): AI-RRM profile name
            freq_band (int): Frequency band enum value (1=2.4 GHz, 2=5 GHz,
                3=6 GHz)
            freq_label (str): Display label (e.g., "5 GHz")

        Returns:
            BuildingMetrics: Populated metrics object

        Raises:
            None: Individual API failures are logged but don't fail the method
        """
        logger.debug(f"  Fetching {freq_label} data...")

        # Initialize metrics object with building metadata
        metrics = BuildingMetrics(
            building_id=floor_uuid,
            building_name=building_name,
            building_hierarchy=building_hierarchy,
            profile_name=profile_name,
            frequency_band=freq_band,
            frequency_label=freq_label,
        )

        # Get coverage data (AP count, client count)
        try:
            coverage = self.client.get_coverage_summary(building_uuid, freq_band)
            if coverage:
                metrics.ap_count = coverage.get("totalApCount", 0)
                metrics.client_count = coverage.get("totalClients", 0)
                metrics.timestamp = coverage.get("timestamp", "")
        except Exception as e:
            logger.warning(f"    Could not fetch coverage data: {e}")

        # Get performance data (RRM health score, changes)
        try:
            performance = self.client.get_performance_summary(building_uuid, freq_band)
            if performance:
                metrics.rrm_health_score = performance.get("rrmHealthScore", 0.0)
                metrics.rrm_changes = performance.get("totalRrmChangesV2", 0)
                if not metrics.timestamp:
                    metrics.timestamp = performance.get("timestamp", "")
        except Exception as e:
            logger.warning(f"    Could not fetch performance data: {e}")

        # Get AI-generated insights (floor-level UUID — insights are floor-scoped)
        try:
            insights = self.client.get_insights(floor_uuid, freq_band)
            metrics.insights = insights
        except Exception as e:
            logger.warning(f"    Could not fetch insights: {e}")

        # Calculate issue status based on collected data
        metrics.calculate_issue_status()

        return metrics

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all collected metrics.

        Aggregates data from all collected building/frequency metrics
        to provide high-level statistics for the executive summary.

        Returns:
            Dict[str, Any]: Summary statistics including:
                - total_buildings: Unique building count
                - buildings_with_issues: Count of buildings flagged
                - total_aps: Sum of all APs
                - total_clients: Sum of all clients
                - total_insights: Sum of all insights
                - average_health_score: Mean health score
                - collection_timestamp: ISO format timestamp
                Returns empty dict if no metrics collected

        Raises:
            None
        """
        # Edge case: No metrics collected
        if not self.metrics:
            return {}

        # Calculate unique building count (multiple frequencies per building)
        total_buildings = len(set(m.building_id for m in self.metrics))
        buildings_with_issues = len(
            set(m.building_id for m in self.metrics if m.has_issues)
        )

        # Aggregate counts across all frequency bands
        total_aps = sum(m.ap_count for m in self.metrics)
        total_clients = sum(m.client_count for m in self.metrics)
        total_insights = sum(len(m.insights) for m in self.metrics)

        # Calculate average health score across all building/band combinations
        avg_health_score = (
            sum(m.rrm_health_score for m in self.metrics) / len(self.metrics)
            if self.metrics
            else 0
        )

        return {
            "total_buildings": total_buildings,
            "buildings_with_issues": buildings_with_issues,
            "total_aps": total_aps,
            "total_clients": total_clients,
            "total_insights": total_insights,
            "average_health_score": round(avg_health_score, 2),
            "collection_timestamp": datetime.now().isoformat(),
        }
