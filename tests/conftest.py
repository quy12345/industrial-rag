"""Keep the offline test suite deterministic and isolated from local secrets."""

from __future__ import annotations

import os

from app.config import Settings, get_settings

# Pytest loads this module before collecting test modules such as app.main. Disable the repository
# .env file and remove matching process variables inside the test process so a developer credential
# can never alter assertions or appear in pytest failure output.
Settings.model_config["env_file"] = None
for field_name in Settings.model_fields:
    os.environ.pop(field_name.upper(), None)
get_settings.cache_clear()
