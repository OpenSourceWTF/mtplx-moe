from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SettingType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


class Visibility(str, Enum):
    PUBLIC = "public"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"
    INTERNAL = "internal"


class Lifecycle(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    COMPATIBILITY = "compatibility"
    RETIRED = "retired"


@dataclass(frozen=True)
class SettingAlias:
    name: str
    source: str
    replacement: str | None = None


@dataclass(frozen=True)
class SettingSpec:
    name: str
    value_type: SettingType
    default: Any
    domain: str = "runtime"
    visibility: Visibility = Visibility.ADVANCED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    description: str = ""
    aliases: tuple[SettingAlias, ...] = field(default_factory=tuple)
    choices: tuple[str, ...] = field(default_factory=tuple)
    minimum: int | float | None = None
    maximum: int | float | None = None
    secret: bool = False
    live_mutable: bool = False
    validator: Callable[[Any], Any] | None = None

    def parse(self, raw: Any) -> Any:
        try:
            if self.value_type is SettingType.BOOL:
                if isinstance(raw, bool):
                    value = raw
                else:
                    text = str(raw).strip().lower()
                    if text in {"1", "true", "yes", "on"}:
                        value = True
                    elif text in {"0", "false", "no", "off"}:
                        value = False
                    else:
                        raise ValueError(f"{self.name}: expected boolean")
            elif self.value_type is SettingType.INT:
                value = int(raw)
            elif self.value_type is SettingType.FLOAT:
                value = float(raw)
            else:
                value = str(raw)
        except (TypeError, ValueError) as exc:
            if str(exc).startswith(f"{self.name}:"):
                raise
            raise ValueError(
                f"{self.name}: expected {self.value_type.value}"
            ) from exc

        if self.choices and value not in self.choices:
            raise ValueError(
                f"{self.name}: expected one of {', '.join(self.choices)}"
            )
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name}: expected >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name}: expected <= {self.maximum}")
        return self.validator(value) if self.validator else value
