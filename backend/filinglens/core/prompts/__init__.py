"""Versioned prompt files. Each file has '# system' and '# user' sections."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str

    def render(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)


def load_prompt(name: str) -> Prompt:
    raw = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    _, _, rest = raw.partition("# system\n")
    system, _, user = rest.partition("# user\n")
    return Prompt(version=name, system=system.strip(), user_template=user.strip())
