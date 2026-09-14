# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Filesystem identities, preserved verbatim rather than sanitized."""

from pathlib import PureWindowsPath


def validate_identifier(value: object, *, field: str) -> str:
    """Require a portable single component and reserve legacy swap suffixes."""
    if (
        not isinstance(value, str) or not value or value in (".", "..", ".build-work")
        or any(character in value for character in ("/", "\\", "\0"))
        or PureWindowsPath(value).drive or value.endswith((".tmp", ".old"))
    ):
        raise ValueError(
            f"{field}: {value!r} must be a nonempty safe path component; absolute paths, separators, "
            "dot components, NULs, .build-work and the reserved .tmp/.old suffixes are not allowed"
        )
    return value
