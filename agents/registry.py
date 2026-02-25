import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List


CONFIG_PATH = Path(__file__).with_name("agents.json")


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    module: str
    runner: str
    experiment: str


def _load_configs() -> List[AgentConfig]:
    data = CONFIG_PATH.read_text()
    entries = json.loads(data)
    configs: List[AgentConfig] = []
    for entry in entries:
        configs.append(AgentConfig(**entry))
    return configs


AGENT_CONFIGS: List[AgentConfig] = _load_configs()


def _load_agent_registry() -> Dict[str, Callable[[str], str]]:
    registry: Dict[str, Callable[[str], str]] = {}
    for agent in AGENT_CONFIGS:
        module = import_module(agent.module)
        runner = getattr(module, agent.runner)
        if not callable(runner):
            raise TypeError(f"Runner '{agent.runner}' in module '{agent.module}' is not callable.")
        registry[agent.agent_id] = runner
    return registry


AGENT_REGISTRY: Dict[str, Callable[[str], str]] = _load_agent_registry()
