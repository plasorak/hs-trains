"""Shared utilities for the rusty-trains Python tooling."""

from functools import cache
from pathlib import Path

_SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "railml/railML-3.3-SR1/source/schema/railml3.xsd"
)
_DCTERMS_STUB = _SCHEMA_PATH.parent / "dcterms_stub.xsd"


@cache
def _load_schema():
    import xmlschema
    return xmlschema.XMLSchema(
        str(_SCHEMA_PATH),
        locations={"http://purl.org/dc/terms/": str(_DCTERMS_STUB)},
    )


def validate_xml(xml_str: str) -> list[str]:
    """Validate a RailML 3.3 XML string against the XSD.

    Returns a list of error message strings, empty if the document is valid.
    """
    if not _SCHEMA_PATH.exists():
        return ["XSD schema not found — cannot validate."]
    try:
        xs = _load_schema()
        return [str(e) for e in xs.iter_errors(xml_str)]
    except Exception as exc:
        return [str(exc)]
