#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
tests/test_search.py
--------------------
Tests for the Search Agent.

Run with:
    pytest tests/test_search.py -v
"""

import pytest
import pandas as pd
import agents.search as search_agent
from agents.search import search_products, load_products


# ── Dataset loading ───────────────────────────────────────────────────────────

def test_load_products_returns_dataframe():
    df = load_products()
    assert len(df) > 0, "Product dataset should not be empty"


def test_load_products_has_required_columns():
    df = load_products()
    required = ["product_id", "product_name", "brand", "category", "store", "price", "mrp", "rating", "stock_status"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


# ── Keyword search ────────────────────────────────────────────────────────────

def test_search_returns_results_for_valid_query():
    results = search_products(query="phone")
    assert len(results) > 0, "Should return results for 'phone'"


def test_search_returns_empty_for_unknown_query():
    results = search_products(query="xyzproductthatdoesnotexist123")
    assert results == [], "Should return empty list for unknown query"


def test_search_is_case_insensitive():
    lower = search_products(query="phone")
    upper = search_products(query="PHONE")
    assert len(lower) == len(upper), "Search should be case insensitive"


# ── Budget filter ─────────────────────────────────────────────────────────────

def test_search_respects_budget():
    results = search_products(query="phone", max_budget=1500)
    for r in results:
        assert r["price"] <= 1500, f"Product price {r['price']} exceeds budget 1500"


def test_search_with_zero_budget_returns_empty():
    results = search_products(query="phone", max_budget=0)
    assert results == [], "Budget of 0 should return no results"


# ── Category filter ───────────────────────────────────────────────────────────

def test_search_filters_by_category():
    results = search_products(query="", category="Electronics")
    for r in results:
        assert r["category"].lower() == "electronics", f"Expected Electronics, got {r['category']}"


def test_search_category_is_case_insensitive():
    lower = search_products(query="", category="electronics")
    upper = search_products(query="", category="Electronics")
    assert len(lower) == len(upper), "Category filter should be case insensitive"


# ── Stock filter ──────────────────────────────────────────────────────────────

def test_search_excludes_out_of_stock():
    results = search_products(query="")
    for r in results:
        assert r["stock_status"] != "out_of_stock", "Out of stock products should be excluded"


# ── Combined filters ──────────────────────────────────────────────────────────

def test_search_combined_query_budget_category():
    results = search_products(query="serum", max_budget=600, category="Electronics")
    for r in results:
        assert r["price"] <= 600
        assert r["category"].lower() == "electronics"


def test_load_products_from_woocommerce_queries_stores_a_and_b_and_maps_products(
    monkeypatch,
):
    key_a, secret_a, key_b, secret_b = object(), object(), object(), object()
    monkeypatch.setattr(search_agent.config, "STORE_A_URL", "http://store-a")
    monkeypatch.setattr(search_agent.config, "WC_CONSUMER_KEY", key_a)
    monkeypatch.setattr(search_agent.config, "WC_CONSUMER_SECRET", secret_a)
    monkeypatch.setattr(search_agent.config, "STORE_B_URL", "http://store-b")
    monkeypatch.setattr(search_agent.config, "STORE_B_CONSUMER_KEY", key_b)
    monkeypatch.setattr(search_agent.config, "STORE_B_CONSUMER_SECRET", secret_b)
    monkeypatch.setattr(search_agent.config, "STORE_C_URL", "http://store-c")
    monkeypatch.setattr(search_agent.config, "STORE_C_CONSUMER_KEY", "key-c")
    monkeypatch.setattr(search_agent.config, "STORE_C_CONSUMER_SECRET", "secret-c")
    calls = []

    def fetch_products(store_url, key, secret):
        calls.append((store_url, key, secret))
        if store_url == "http://store-b":
            return [
                {
                    "id": 202,
                    "name": "Store B Phone",
                    "price": "24999",
                    "regular_price": "29999",
                    "stock_status": "instock",
                    "short_description": "RAM: 8 GB",
                    "meta_data": [
                        {"key": "dsa_brand", "value": "Example"},
                        {"key": "dsa_store", "value": "Store B"},
                        {"key": "dsa_rating", "value": "4.5"},
                        {"key": "dsa_review_count", "value": "120"},
                    ],
                }
            ]
        return []

    monkeypatch.setattr(search_agent, "_fetch_store_products", fetch_products)

    products = search_agent.load_products_from_woocommerce()

    assert calls == [
        ("http://store-a", key_a, secret_a),
        ("http://store-b", key_b, secret_b),
    ]
    assert len(products) == 1
    assert products.iloc[0].to_dict() == {
        "product_id": "202",
        "product_name": "Store B Phone",
        "brand": "Example",
        "category": "",
        "subcategory": "",
        "store": "Store B",
        "price": 24999.0,
        "mrp": 29999.0,
        "rating": 4.5,
        "review_count": 120,
        "stock_status": "in_stock",
        "key_specs": "RAM: 8 GB",
        "pros": "",
        "cons": "",
        "spec_confidence": "low",
        "spec_ram": 0.0,
        "spec_storage": 0.0,
        "spec_battery": 0.0,
        "spec_rear_camera": 0.0,
        "spec_display_size": 0.0,
    }


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("STORE_B_URL", ""),
        ("STORE_B_CONSUMER_KEY", ""),
        ("STORE_B_CONSUMER_SECRET", ""),
    ],
)
def test_load_products_from_woocommerce_skips_store_b_without_complete_config(
    monkeypatch, setting, value
):
    monkeypatch.setattr(search_agent.config, "STORE_A_URL", "http://store-a")
    monkeypatch.setattr(search_agent.config, "WC_CONSUMER_KEY", object())
    monkeypatch.setattr(search_agent.config, "WC_CONSUMER_SECRET", object())
    monkeypatch.setattr(search_agent.config, "STORE_B_URL", "http://store-b")
    monkeypatch.setattr(search_agent.config, "STORE_B_CONSUMER_KEY", object())
    monkeypatch.setattr(search_agent.config, "STORE_B_CONSUMER_SECRET", object())
    monkeypatch.setattr(search_agent.config, setting, value)
    calls = []
    monkeypatch.setattr(
        search_agent,
        "_fetch_store_products",
        lambda store_url, key, secret: calls.append(store_url) or [],
    )

    search_agent.load_products_from_woocommerce()

    assert calls == ["http://store-a"]


def test_load_products_falls_back_to_csv_when_woocommerce_is_empty(monkeypatch):
    csv_products = pd.DataFrame([{"product_name": "CSV Phone", "store": "Store B"}])
    monkeypatch.setattr(
        search_agent, "load_products_from_woocommerce", lambda: pd.DataFrame()
    )
    monkeypatch.setattr(
        search_agent, "load_products_from_csv", lambda: csv_products
    )

    products = search_agent.load_products()

    pd.testing.assert_frame_equal(products, csv_products)
