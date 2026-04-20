"""
API Client for DNA Center AI-RRM data collection.

This module provides a client interface for interacting with Cisco DNA
Center APIs, specifically for AI-RRM (AI Radio Resource Management)
data collection using both REST and GraphQL endpoints.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from auth import DNACenterAuth

logger = logging.getLogger(__name__)


class DNACenterClient:
    """
    Client for interacting with DNA Center APIs.

    This class provides methods to query AI-RRM data including building
    configurations, coverage metrics, performance statistics, and
    AI-generated insights via GraphQL queries.
    """

    def __init__(self, auth: DNACenterAuth, max_retries: int = 3) -> None:
        """
        Initialize DNA Center API client with retry logic.

        Parameters:
            auth (DNACenterAuth): Authenticated DNA Center session
            max_retries (int): Maximum number of retry attempts for failed requests

        Returns:
            None
        """
        self.auth: DNACenterAuth = auth
        self.base_url: str = auth.base_url
        self.verify_ssl: bool = auth.verify_ssl
        self.max_retries: int = max_retries

        # Lazily populated by _get_site_map(); avoids an extra API call on
        # init and allows health_check() to succeed before the map is needed.
        self._site_map: Optional[Dict[str, Dict[str, Any]]] = None

        # Thread-local storage for requests.Session objects.
        # requests.Session is not thread-safe, so each thread gets its own.
        self._local = threading.local()

        logger.debug(f"API client initialized with {max_retries} max retries")

    def _get_session(self) -> requests.Session:
        """
        Return a thread-local requests.Session with retry strategy.

        Each thread gets its own Session instance to avoid
        concurrency issues with shared cookie jars and connection
        pool state.

        Returns:
            requests.Session: Configured session for the current thread
        """
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        """
        Make authenticated request to DNA Center API with retry logic.

        This is a private helper method that handles authentication
        headers, SSL verification, retries, and error handling for all API calls.

        Parameters:
            method (str): HTTP method (GET, POST, PUT, DELETE)
            endpoint (str): API endpoint path (e.g., '/api/v1/health')
            **kwargs (Any): Additional arguments passed to requests library

        Returns:
            requests.Response: HTTP response object

        Raises:
            requests.exceptions.RequestException: For HTTP errors,
                connection issues, or timeouts
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.auth.get_headers()

        # Merge additional headers if provided
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        try:
            response = self._get_session().request(
                method=method,
                url=url,
                headers=headers,
                verify=self.verify_ssl,
                timeout=60,
                **kwargs,
            )
            # Raise exception for 4xx/5xx status codes
            response.raise_for_status()
            logger.debug(f"API request successful: {method} {endpoint}")
            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"API request timeout: {method} {endpoint} - {e}")
            raise

        except requests.exceptions.ConnectionError as e:
            logger.error(f"API connection error: {method} {endpoint} - {e}")
            logger.error(f"Check that Catalyst Center is reachable at {self.base_url}")
            raise

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"API HTTP error: {method} {endpoint} - Status {e.response.status_code}"
            )
            if e.response.status_code == 401:
                logger.error("Authentication failed - check credentials")
            elif e.response.status_code == 403:
                logger.error("Access forbidden - check user permissions")
            elif e.response.status_code == 404:
                logger.error(f"Endpoint not found: {endpoint}")
            raise

        except requests.exceptions.RequestException as e:
            # Log error with full context for troubleshooting
            logger.error(f"API request failed: {method} {endpoint} - {e}")
            raise

    def health_check(self) -> bool:
        """
        Perform health check to verify connectivity to Catalyst Center.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            logger.info(f"Performing health check: {self.base_url}")

            # Try a simple API endpoint to verify connectivity
            response = self._make_request(
                "GET", "/api/v1/dna/sunray/airfprofilesitesinfo"
            )

            logger.info("✓ Health check passed - Catalyst Center is reachable")
            return True

        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Health check failed - Cannot reach {self.base_url}")
            logger.error("  Check network connectivity and Catalyst Center URL")
            return False

        except requests.exceptions.Timeout:
            logger.error(f"✗ Health check failed - Request timeout to {self.base_url}")
            logger.error("  Catalyst Center may be overloaded or network is slow")
            return False

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("✗ Health check failed - Authentication error")
                logger.error("  Check username and password in .env file")
            else:
                logger.error(f"✗ Health check failed - HTTP {e.response.status_code}")
            return False

        except Exception as e:
            logger.error(f"✗ Health check failed - Unexpected error: {e}")
            return False

    # Maximum sites per page for /dna/intent/api/v1/site pagination.
    _SITE_PAGE_LIMIT: int = 500

    def _get_site_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Return cached site map, fetching all pages from API on first call.

        The /dna/intent/api/v1/site endpoint silently caps results at
        500 records per request. This method paginates with offset/limit
        to ensure all sites are retrieved on large deployments.

        Parameters:
            None

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of site UUID to site metadata
                with keys: name, type, parentId, hierarchy
        """
        if self._site_map is not None:
            return self._site_map

        logger.info("Building site map from /dna/intent/api/v1/site")
        self._site_map = {}
        offset = 1  # API uses 1-based indexing

        while True:
            try:
                response = self._make_request(
                    "GET",
                    "/dna/intent/api/v1/site",
                    params={"offset": offset, "limit": self._SITE_PAGE_LIMIT},
                )
                sites = response.json().get("response", [])
            except Exception as e:
                logger.warning(
                    f"Could not fetch site map (offset={offset}): {e}. "
                    "Building UUID resolution may be incomplete."
                )
                break

            for site in sites:
                site_type: Optional[str] = None
                for info in site.get("additionalInfo", []):
                    if info.get("nameSpace") == "Location":
                        site_type = info.get("attributes", {}).get("type")
                self._site_map[site["id"]] = {
                    "name": site.get("name", ""),
                    "type": site_type,
                    "parentId": site.get("parentId", ""),
                    "hierarchy": site.get("siteNameHierarchy", ""),
                }

            if len(sites) < self._SITE_PAGE_LIMIT:
                break
            offset += self._SITE_PAGE_LIMIT

        logger.info(f"Site map built: {len(self._site_map)} sites indexed")
        return self._site_map

    def resolve_building_uuid(self, site_uuid: str) -> str:
        """
        Return the building-level UUID for a given site UUID.

        The AI-RRM profile API returns floor-level UUIDs in its
        associatedBuildings array. GraphQL coverage and performance queries
        require a building-level UUID. This method walks the parentId chain
        to find the correct building UUID.

        Parameters:
            site_uuid (str): UUID from the AI-RRM profile site list

        Returns:
            str: Building-level UUID. Returns site_uuid unchanged if it is
                already a building, is unknown, or the site map is unavailable.
        """
        site_map = self._get_site_map()
        site = site_map.get(site_uuid, {})

        site_type = site.get("type")
        if not site or site_type in ("building", None):
            return site_uuid

        if site_type == "floor":
            parent_id = site.get("parentId", "")
            parent = site_map.get(parent_id, {})
            if parent_id and parent.get("type") == "building":
                logger.debug(f"Resolved floor {site_uuid} → building {parent_id}")
                return parent_id

        logger.debug(
            f"Could not resolve building for site {site_uuid} (type={site_type}). "
            f"Using original UUID."
        )
        return site_uuid

    def get_airrm_buildings(self) -> List[Dict[str, Any]]:
        """
        Get list of buildings with AI-RRM enabled (building-level only).

        Queries the DNA Center API to retrieve all buildings that have
        AI-RRM profiles configured. The API returns floor-level sites,
        so this method groups them by building name to return one entry
        per building, matching AI-RRM's building-level operation model.

        Returns:
            List[Dict[str, Any]]: List of building dictionaries, each
                containing building metadata (UUID, name, hierarchy)
                and AI-RRM profile information. Deduplicated to show
                one entry per building.

        Raises:
            requests.exceptions.RequestException: If API call fails

        Example:
            >>> client.get_airrm_buildings()
            [{'instanceUUID': 'abc-123', 'name': 'Building 1', ...}]
        """
        logger.info("Fetching AI-RRM enabled buildings")
        endpoint = "/api/v1/dna/sunray/airfprofilesitesinfo"

        response = self._make_request("GET", endpoint)
        data = response.json()

        # Use dict to deduplicate floor-level entries by building name
        building_map: Dict[str, Dict[str, Any]] = {}
        floor_count = 0

        # Parse response and extract buildings from each profile
        if "response" in data:
            for profile in data["response"]:
                profile_name = profile.get("aiRfProfileName", "Unknown")
                for site in profile.get("associatedBuildings", []):
                    floor_count += 1
                    building_name = site.get("name", "")

                    # Group by building name - keep first occurrence
                    # AI-RRM operates at building level, not floor level
                    if building_name and building_name not in building_map:
                        site["aiRfProfileName"] = profile_name
                        building_map[building_name] = site
                        logger.debug(
                            f"Added building: {building_name} "
                            f"(UUID: {site.get('instanceUUID')})"
                        )

        buildings = list(building_map.values())

        logger.info(
            f"Found {len(buildings)} buildings with AI-RRM enabled "
            f"(deduplicated from {floor_count} floor-level sites)"
        )
        return buildings

    def graphql_query(
        self, operation_name: str, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query against DNA Center.

        Parameters:
            operation_name (str): Name of the GraphQL operation
            query (str): GraphQL query string
            variables (Dict[str, Any]): Query variables dictionary

        Returns:
            Dict[str, Any]: Query response data

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        endpoint = (
            "/api/kairos/v1/proxy/api/v2/core-services/customer-id/sunray/graphql"
        )

        payload = {
            "operationName": operation_name,
            "variables": variables,
            "query": query,
        }

        logger.debug(f"Executing GraphQL query: {operation_name}")
        response = self._make_request("POST", endpoint, json=payload)
        return response.json()

    def get_coverage_summary(
        self, building_id: str, frequency_band: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get RF coverage summary for a building and frequency band.

        Retrieves coverage metrics including AP count, client count,
        and SNR data for a specific building and frequency.

        Parameters:
            building_id (str): Building UUID
            frequency_band (int): Frequency band (2=2.4GHz, 5=5GHz,
                6=6GHz)

        Returns:
            Optional[Dict[str, Any]]: Coverage summary data with keys:
                - totalApCount: Number of access points
                - totalClients: Number of connected clients
                - connectivitySnr: SNR metrics
                Returns None if no data available

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        query = """query getRfCoverageSummaryLatest01(
            $buildingId: String,
            $frequencyBand: Int
        ) {
            getRfCoverageSummaryLatest01(
                buildingId: $buildingId,
                frequencyBand: $frequencyBand
            ) {
                nodes {
                    buildingId
                    frequencyBand
                    siteId
                    timestampMs
                    timestamp
                    connectivitySnr
                    connectivitySnrDensity
                    apDensity
                    totalApCount
                    totalClients
                }
            }
        }"""

        variables = {"buildingId": building_id, "frequencyBand": frequency_band}

        result = self.graphql_query("getRfCoverageSummaryLatest01", query, variables)

        # Extract first node from result (edge case: empty response)
        nodes = (
            result.get("data", {})
            .get("getRfCoverageSummaryLatest01", {})
            .get("nodes", [])
        )
        return nodes[0] if nodes else None

    def get_performance_summary(
        self, building_id: str, frequency_band: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get RF performance summary for a building and frequency band.

        Retrieves RRM performance metrics including health score,
        optimization changes, and CCI (co-channel interference) data.

        Parameters:
            building_id (str): Building UUID
            frequency_band (int): Frequency band (2=2.4GHz, 5=5GHz,
                6=6GHz)

        Returns:
            Optional[Dict[str, Any]]: Performance summary with keys:
                - rrmHealthScore: Health score (0-100)
                - totalRrmChangesV2: Number of RRM optimizations
                - apPercentageWithHighCci: % APs with high interference
                Returns None if no data available

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        query = """query getRfPerformanceSummaryLatest01(
            $buildingId: String,
            $frequencyBand: Int
        ) {
            getRfPerformanceSummaryLatest01(
                buildingId: $buildingId,
                frequencyBand: $frequencyBand
            ) {
                nodes {
                    buildingId
                    frequencyBand
                    siteId
                    timestampMs
                    timestamp
                    totalRrmChangesV2
                    rrmHealthScore
                    apPercentageWithHighCci
                }
            }
        }"""

        variables = {"buildingId": building_id, "frequencyBand": frequency_band}

        result = self.graphql_query("getRfPerformanceSummaryLatest01", query, variables)

        # Edge case: Handle missing or empty response
        nodes = (
            result.get("data", {})
            .get("getRfPerformanceSummaryLatest01", {})
            .get("nodes", [])
        )
        return nodes[0] if nodes else None

    def get_insights(
        self, building_id: str, frequency_band: int
    ) -> List[Dict[str, Any]]:
        """
        Get current AI-generated insights for a building/frequency.

        Retrieves AI-RRM insights which are recommendations and
        observations about the RF environment.

        Parameters:
            building_id (str): Building UUID
            frequency_band (int): Frequency band (2=2.4GHz, 5=5GHz,
                6=6GHz)

        Returns:
            List[Dict[str, Any]]: List of insight dictionaries with:
                - insightType: Type/category of insight
                - insightValue: Numeric value associated with insight
                - description: Human-readable description
                - reason: Explanation of why insight was generated
                Returns empty list if no insights available

        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        query = """query getCurrentInsights01(
            $buildingId: String,
            $frequencyBand: Int
        ) {
            getCurrentInsights01(
                buildingId: $buildingId,
                frequencyBand: $frequencyBand
            ) {
                nodes {
                    buildingId
                    frequencyBand
                    siteId
                    timestampMs
                    timestamp
                    insightType
                    insightValue
                    description
                    reason
                }
            }
        }"""

        variables = {"buildingId": building_id, "frequencyBand": frequency_band}

        result = self.graphql_query("getCurrentInsights01", query, variables)

        # Edge case: Return empty list instead of None for consistency
        nodes = result.get("data", {}).get("getCurrentInsights01", {}).get("nodes", [])
        return nodes if nodes else []
