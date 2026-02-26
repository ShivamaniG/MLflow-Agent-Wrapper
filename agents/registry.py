import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List


CONFIG_PATH = Path(__file__).with_name("agents.json")


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    experiment: str


def _load_configs() -> List[AgentConfig]:
    data = CONFIG_PATH.read_text()
    entries = json.loads(data)
    return [AgentConfig(agent_id=entry["agent_id"], experiment=entry["experiment"]) for entry in entries]


AGENT_CONFIGS: List[AgentConfig] = _load_configs()


def _load_agent_registry() -> Dict[str, Callable[[str], str]]:
    registry: Dict[str, Callable[[str], str]] = {}
    for agent in AGENT_CONFIGS:
        module_name = f"agents.{agent.agent_id}_agent"
        runner_name = f"run_{agent.agent_id}_agent"
        module = import_module(module_name)
        runner = getattr(module, runner_name)
        if not callable(runner):
            raise TypeError(f"Runner '{runner_name}' in module '{module_name}' is not callable.")
        registry[agent.agent_id] = runner
    return registry


AGENT_REGISTRY: Dict[str, Callable[[str], str]] = _load_agent_registry()
