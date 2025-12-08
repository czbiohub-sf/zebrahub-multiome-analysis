# core_http.py
import time, logging, functools
import requests, requests_cache, backoff
from ratelimit import limits

from scripts.litemind_peak_analysis.config import CACHE_DIR
CACHE_DIR.mkdir(exist_ok=True)
requests_cache.install_cache(str(CACHE_DIR / "litebio"), expire_after=86400)  # one-day cache

def _log_retry(details):
    logging.warning("Retrying %s after %s tries", details['args'][0].url,
                    details['tries'])

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    max_tries=5,
    on_backoff=_log_retry,
)
@limits(calls=5, period=1)          # polite default: ≤ 5 req s-1
def _safe_get(url, **kw):
    r = requests.get(url, timeout=10, **kw)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type","").startswith("application/json") else r.text

def fetch_json(url, **params):
    try:
        # if one of the params is 'content_type', replace it by 'content-type':
        params = {k.replace('_', '-'): v for k, v in params.items()}

        return _safe_get(url, params=params)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        return f"Error: {type(e).__name__} with message: '{str(e)}' occurred while trying to fetch JSON from {url} with params {params}."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {type(e).__name__} with message: '{str(e)}' occurred while trying to fetch JSON from {url} with params {params}."
