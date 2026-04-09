"""Utilities to normalize and serialize product text for embedding."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


class TextNormalizer:
    """Normalize textual fields and build a single sentence representation."""

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        normalized = unicodedata.normalize("NFD", text)
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def normalize_product_text(self, product_data: dict[str, Any]) -> dict[str, Any]:
        additional_props = self._normalize_additional_properties(
            product_data.get("additionalProperties", [])
        )

        color = self.normalize(str(product_data.get("color", "") or ""))
        if not color:
            color = self._extract_color_from_properties(additional_props)

        description = str(product_data.get("description", "") or "")
        if len(description) > 2500:
            description = description[:2500]

        return {
            "name": self.normalize(str(product_data.get("name", "") or "")),
            "brand": self._normalize_brand(product_data.get("brand")),
            "sku": self.normalize(str(product_data.get("sku", "") or "")),
            "mpn": self.normalize(str(product_data.get("mpn", "") or "")),
            "additionalProperties": additional_props,
            "description": self.normalize(description),
            "color": color,
        }

    def build_sentence(self, product_data: dict[str, Any]) -> str:
        parts: list[str] = []

        name = product_data.get("name") or ""
        brand = product_data.get("brand") or ""
        mpn = product_data.get("mpn") or ""
        color = product_data.get("color") or ""
        description = product_data.get("description") or ""
        additional_props = product_data.get("additionalProperties") or []

        if name:
            parts.append(name)
        if brand:
            parts.append(f"marca {brand}")
        if mpn:
            parts.append(f"mpn {mpn}")
        if color:
            parts.append(f"cor {color}")
        if description:
            parts.append(description)

        for prop in additional_props:
            if not isinstance(prop, dict):
                continue
            prop_name = (prop.get("name") or "").strip()
            prop_value = (prop.get("value") or "").strip()
            if not prop_name and not prop_value:
                continue
            if prop_name and prop_value:
                parts.append(f"{prop_name} {prop_value}")
            elif prop_value:
                parts.append(prop_value)

        return ", ".join(parts)

    def normalize_reference_product(self, product: Any) -> dict[str, Any]:
        additional_props = []
        for prop in getattr(product, "additional_properties", []) or []:
            additional_props.append(
                {
                    "name": getattr(prop, "name", "") or "",
                    "value": getattr(prop, "value", "") or "",
                }
            )

        brand_name = ""
        brand = getattr(product, "brand", None)
        if brand is not None:
            brand_name = getattr(brand, "name", "") or ""

        return self.normalize_product_text(
            {
                "name": getattr(product, "name", "") or "",
                "brand": {"name": brand_name},
                "sku": getattr(product, "sku", "") or "",
                "mpn": getattr(product, "mpn", "") or "",
                "description": getattr(product, "description", "") or "",
                "color": getattr(product, "color", "") or "",
                "additionalProperties": additional_props,
            }
        )

    def build_reference_product_sentence(self, product: Any) -> str:
        normalized = self.normalize_reference_product(product)
        return self.build_sentence(normalized)

    def _normalize_brand(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, dict):
            return self.normalize(str(raw.get("name", "") or ""))
        if isinstance(raw, str):
            return self.normalize(raw)
        return ""

    def _normalize_additional_properties(self, raw: Any) -> list[dict[str, str]]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            return []

        normalized_props: list[dict[str, str]] = []
        for prop in raw:
            if isinstance(prop, dict):
                normalized_props.append(
                    {
                        "name": self.normalize(str(prop.get("name", "") or "")),
                        "value": self.normalize(str(prop.get("value", "") or "")),
                    }
                )
        return normalized_props

    def _extract_color_from_properties(
        self,
        additional_properties: list[dict[str, str]],
    ) -> str:
        if not additional_properties:
            return ""

        color_keys = ["cor", "color", "colour", "cor principal", "cor do produto"]

        for prop in additional_properties:
            name = (prop.get("name", "") or "").lower()
            if any(key in name for key in color_keys):
                return prop.get("value", "") or ""

        return ""
