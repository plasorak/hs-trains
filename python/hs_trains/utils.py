"""Shared utilities for the hs-trains Python tooling."""

from functools import cache
from pathlib import Path

# Expected location for the RailML 3.3 XSD schema.  The schema is not bundled
# (it is CC BY-NC-ND 4.0); see the README for download instructions.
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
    Raises FileNotFoundError if the XSD schema file is not present.
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"RailML XSD not found at {_SCHEMA_PATH}. "
            "Download the RailML 3.3 schema (non-commercial use only) from "
            "https://www.railml.org/en/download/ and extract it so that "
            "railml/railML-3.3-SR1/source/schema/railml3.xsd exists at the repo root."
        )
    try:
        xs = _load_schema()
        return [str(e) for e in xs.iter_errors(xml_str)]
    except Exception as exc:
        return [str(exc)]
