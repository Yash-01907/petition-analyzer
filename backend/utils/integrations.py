# utils/integrations.py — External platform adapter pattern.
# Phase 9: Post-MVP Enhancement #2.
#
# Provides a pluggable interface for importing campaign data from
# ActionKit, NationBuilder, or any future CRM/advocacy platform.
# Currently implements stub adapters — real implementations would
# use each platform's REST API with OAuth credentials.

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class PlatformAdapter(ABC):
    """Base class for external advocacy platform integrations.

    Subclass this to add support for new platforms. Each adapter must
    implement fetch_campaigns() which returns a standardized DataFrame.
    """

    @abstractmethod
    def authenticate(self, api_key: str, **kwargs) -> bool:
        """Authenticate with the platform API.

        Args:
            api_key: API key or token for the platform.

        Returns:
            True if authentication succeeds.
        """
        ...

    @abstractmethod
    def fetch_campaigns(
        self,
        limit: Optional[int] = None,
        since: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch campaign data from the platform.

        Must return a DataFrame with at minimum these columns:
            headline, body_text, cta_text, unique_visitors,
            signatures, traffic_source

        Args:
            limit: Maximum number of campaigns to fetch.
            since: ISO date string — only fetch campaigns after this date.

        Returns:
            Standardized DataFrame ready for the ingestion pipeline.
        """
        ...

    @abstractmethod
    def platform_name(self) -> str:
        """Return the human-readable name of this platform."""
        ...


class ActionKitAdapter(PlatformAdapter):
    """Adapter for ActionKit advocacy platform.

    ActionKit's REST API provides campaign/action data via
    /rest/v1/action/ and /rest/v1/page/ endpoints.

    Production implementation would use:
        - Basic Auth or API key auth
        - Paginated GET /rest/v1/action/?_limit=100
        - Map ActionKit's 'title' → headline, 'content' → body_text
    """

    def __init__(self):
        self._authenticated = False
        self._base_url = ""
        self._api_key = ""

    def platform_name(self) -> str:
        return "ActionKit"

    def authenticate(self, api_key: str, base_url: str = "", **kwargs) -> bool:
        """Authenticate with ActionKit API.

        Args:
            api_key: ActionKit API key.
            base_url: ActionKit instance URL (e.g. https://act.example.org).
        """
        # In production: validate credentials with a test API call
        # requests.get(f"{base_url}/rest/v1/action/?_limit=1",
        #              headers={"Authorization": f"Bearer {api_key}"})
        self._api_key = api_key
        self._base_url = base_url
        self._authenticated = True
        return True

    def fetch_campaigns(
        self,
        limit: Optional[int] = None,
        since: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch petition pages from ActionKit.

        Production implementation:
            1. GET /rest/v1/petitionpage/?_limit={limit}
            2. For each page, GET /rest/v1/petitionaction/?page={page_id}
            3. Map fields: title→headline, intro_text→body_text,
               petition_text→cta_text, total_actions→signatures
        """
        if not self._authenticated:
            raise ConnectionError("Not authenticated. Call authenticate() first.")

        # Stub: return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            "headline", "body_text", "cta_text",
            "unique_visitors", "signatures", "traffic_source",
            "cause_category", "campaign_duration_days",
            "has_image", "has_video",
        ])


class NationBuilderAdapter(PlatformAdapter):
    """Adapter for NationBuilder advocacy platform.

    NationBuilder's API v2 provides petition data via
    /api/v2/petitions and /api/v2/petition_signatures endpoints.

    Production implementation would use:
        - OAuth2 bearer token auth
        - GET /api/v2/petitions?limit=100
        - Map NationBuilder fields to the standard schema
    """

    def __init__(self):
        self._authenticated = False
        self._slug = ""
        self._token = ""

    def platform_name(self) -> str:
        return "NationBuilder"

    def authenticate(self, api_key: str, slug: str = "", **kwargs) -> bool:
        """Authenticate with NationBuilder API.

        Args:
            api_key: NationBuilder API token.
            slug: Nation slug (e.g. 'myorganization').
        """
        # In production: validate with GET /api/v1/sites
        self._token = api_key
        self._slug = slug
        self._authenticated = True
        return True

    def fetch_campaigns(
        self,
        limit: Optional[int] = None,
        since: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch petitions from NationBuilder.

        Production implementation:
            1. GET /api/v2/petitions?per_page={limit}
            2. For each petition, get signature count
            3. Map: name→headline, intro→body_text,
               petition_text→cta_text, signatures_count→signatures
        """
        if not self._authenticated:
            raise ConnectionError("Not authenticated. Call authenticate() first.")

        return pd.DataFrame(columns=[
            "headline", "body_text", "cta_text",
            "unique_visitors", "signatures", "traffic_source",
            "cause_category", "campaign_duration_days",
            "has_image", "has_video",
        ])


# Registry of available adapters
PLATFORM_ADAPTERS = {
    "actionkit": ActionKitAdapter,
    "nationbuilder": NationBuilderAdapter,
}


def get_adapter(platform: str) -> PlatformAdapter:
    """Get a platform adapter by name.

    Args:
        platform: Platform identifier ('actionkit' or 'nationbuilder').

    Returns:
        An instance of the appropriate PlatformAdapter.

    Raises:
        ValueError: If the platform is not supported.
    """
    adapter_class = PLATFORM_ADAPTERS.get(platform.lower())
    if not adapter_class:
        supported = ", ".join(PLATFORM_ADAPTERS.keys())
        raise ValueError(
            f"Unsupported platform: '{platform}'. "
            f"Supported platforms: {supported}"
        )
    return adapter_class()
