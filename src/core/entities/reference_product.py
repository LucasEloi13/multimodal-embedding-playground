from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Brand:
	name: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "Brand | None":
		if not data:
			return None
		return cls(name=data.get("name"))


@dataclass(slots=True)
class NamedLink:
	name: str | None = None
	url: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "NamedLink | None":
		if not data:
			return None
		return cls(name=data.get("name"), url=data.get("url"))


@dataclass(slots=True)
class ImageAsset:
	url: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "ImageAsset | None":
		if not data:
			return None
		return cls(url=data.get("url"))


@dataclass(slots=True)
class VariantAttribute:
	name: str | None = None
	value: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "VariantAttribute | None":
		if not data:
			return None
		return cls(name=data.get("name"), value=data.get("value"))


@dataclass(slots=True)
class Variant:
	price: str | None = None
	regular_price: str | None = None
	sku: str | None = None
	attributes: list[VariantAttribute] = field(default_factory=list)

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "Variant | None":
		if not data:
			return None
		return cls(
			price=data.get("price"),
			regular_price=data.get("regularPrice"),
			sku=data.get("sku"),
			attributes=_build_list(data.get("attributes"), VariantAttribute),
		)


@dataclass(slots=True)
class AdditionalProperty:
	name: str | None = None
	value: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "AdditionalProperty | None":
		if not data:
			return None
		return cls(name=data.get("name"), value=data.get("value"))


@dataclass(slots=True)
class ReferenceProduct:
	url: str | None = None
	canonical_url: str | None = None
	name: str | None = None
	description: str | None = None
	sku: str | None = None
	availability: str | None = None
	brand: Brand | None = None
	breadcrumbs: list[NamedLink] = field(default_factory=list)
	images: list[ImageAsset] = field(default_factory=list)
	main_image: ImageAsset | None = None
	currency: str | None = None
	currency_raw: str | None = None
	price: str | None = None
	cash_price: str | None = None
	regular_price: str | None = None
	seller_name: str | None = None
	other_sellers: list[Any] = field(default_factory=list)
	variants: list[Variant] = field(default_factory=list)
	additional_properties: list[AdditionalProperty] = field(default_factory=list)
	features: list[str] = field(default_factory=list)
	mpn: str | None = None
	material: str | None = None
	color: str | None = None

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> "ReferenceProduct":
		return cls(
			url=data.get("url"),
			canonical_url=data.get("canonicalUrl"),
			name=data.get("name"),
			description=data.get("description"),
			sku=data.get("sku"),
			availability=data.get("availability"),
			brand=Brand.from_dict(data.get("brand")),
			breadcrumbs=_build_list(data.get("breadcrumbs"), NamedLink),
			images=_build_list(data.get("images"), ImageAsset),
			main_image=ImageAsset.from_dict(data.get("mainImage")),
			currency=data.get("currency"),
			currency_raw=data.get("currencyRaw"),
			price=data.get("price"),
			cash_price=data.get("cashPrice"),
			regular_price=data.get("regularPrice"),
			seller_name=data.get("sellerName"),
			other_sellers=list(data.get("otherSellers") or []),
			variants=_build_list(data.get("variants"), Variant),
			additional_properties=_build_list(
				data.get("additionalProperties"),
				AdditionalProperty,
			),
			features=list(data.get("features") or []),
			mpn=data.get("mpn"),
			material=data.get("material"),
			color=data.get("color"),
		)


def _build_list(items: Any, entity_cls: type[Any]) -> list[Any]:
	if not isinstance(items, list):
		return []

	entities: list[Any] = []
	for item in items:
		if not isinstance(item, dict):
			continue

		entity = entity_cls.from_dict(item)
		if entity is not None:
			entities.append(entity)

	return entities
