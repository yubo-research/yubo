import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

SERPAPI_URL = "https://serpapi.com/search.json"


def fetch_json(params: Dict[str, str], timeout_seconds: int = 30) -> Dict:
    response = requests.get(SERPAPI_URL, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def select_best_place(query: str, candidates: List[Dict]) -> Optional[Dict]:
    if not candidates:
        return None
    normalized_query = normalize_query(query)
    best = None
    best_score = -1
    for item in candidates:
        title = item.get("title") or item.get("name") or ""
        address = item.get("address") or ""
        combined = f"{title} {address}"
        normalized_combined = normalize_query(combined)
        score = 0
        if "bites" in normalized_combined:
            score += 2
        if "valentine" in normalized_combined:
            score += 2
        if "bronx" in normalized_combined:
            score += 2
        if normalized_query in normalized_combined:
            score += 4
        if score > best_score:
            best = item
            best_score = score
    return best


def find_place_data_id(query: str, api_key: str) -> Tuple[str, str]:
    params = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": api_key,
    }
    data = fetch_json(params)
    candidates: List[Dict] = []
    if isinstance(data.get("local_results"), list):
        candidates.extend(data["local_results"])
    place_results = data.get("place_results")
    if isinstance(place_results, dict):
        candidates.append(place_results)
    best = select_best_place(query, candidates)
    if not best:
        raise RuntimeError("No matching place found")
    data_id = best.get("data_id") or best.get("data_id_token") or best.get("cid")
    title = best.get("title") or best.get("name") or ""
    if not data_id:
        raise RuntimeError("Found place but missing data_id")
    return str(data_id), str(title)


def iter_reviews(data_id: str, api_key: str, max_reviews: Optional[int] = None, sleep_seconds: float = 2.0) -> Iterable[Dict]:
    fetched = 0
    next_page_token: Optional[str] = None
    while True:
        params: Dict[str, str] = {
            "engine": "google_maps_reviews",
            "data_id": data_id,
            "hl": "en",
            "gl": "us",
            "api_key": api_key,
        }
        if next_page_token:
            params["next_page_token"] = next_page_token
        data = fetch_json(params)
        reviews = data.get("reviews") or []
        for review in reviews:
            yield review
            fetched += 1
            if max_reviews is not None and fetched >= max_reviews:
                return
        pagination = data.get("serpapi_pagination") or {}
        next_page_token = pagination.get("next_page_token")
        if not next_page_token:
            return
        time.sleep(sleep_seconds)


def dump_reviews(reviews: List[Dict], out_path: Optional[str], fmt: Optional[str]) -> None:
    if out_path is None:
        for review in reviews:
            sys.stdout.write(json.dumps(review, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    if fmt is None:
        if out_path.lower().endswith(".jsonl") or out_path.lower().endswith(".ndjson"):
            fmt = "jsonl"
        else:
            fmt = "json"
    if fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for review in reviews:
                f.write(json.dumps(review, ensure_ascii=False))
                f.write("\n")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Bites on Valentine, Bronx, NY")
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", choices=["json", "jsonl"], default=None)
    parser.add_argument("--api-key", default=os.getenv("SERPAPI_API_KEY"))
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY; set env var or pass --api-key")

    data_id, place_title = find_place_data_id(args.query, args.api_key)
    reviews: List[Dict] = []
    for review in iter_reviews(data_id=data_id, api_key=args.api_key, max_reviews=args.max, sleep_seconds=args.sleep):
        reviews.append(review)
    dump_reviews(reviews, args.out, args.format)


if __name__ == "__main__":
    main()
