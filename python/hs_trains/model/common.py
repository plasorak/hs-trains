"""Shared RailML 3.3 namespace constants, base class, and common elements.

Imported by both rollingstock.py and infrastructure.py so that neither
sub-schema model depends on the other.
"""

import uuid
from typing import Optional

from pydantic_xml import BaseXmlModel, attr

# RailML 3.3 namespace URI and prefix alias.
NS = "https://www.railml.org/schemas/3.3"
_NS = "rail3"
_NSMAP = {_NS: NS}


class _Base(BaseXmlModel, nsmap=_NSMAP):
    """Base class that propagates the railML namespace map to all submodels."""


def _make_id() -> str:
    return f"id_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Common identifier elements  (generic3.xsd: Designator, Name)
# Used in both rollingstock and infrastructure sub-schemas.
# ---------------------------------------------------------------------------


class Designator(_Base, tag="designator", ns=_NS):
    register_name: str = attr(name="register")
    entry: str = attr(name="entry")
    description: Optional[str] = attr(name="description", default=None)


class Name(_Base, tag="name", ns=_NS):
    name: str = attr(name="name")
    language: str = attr(name="language")
    description: Optional[str] = attr(name="description", default=None)
