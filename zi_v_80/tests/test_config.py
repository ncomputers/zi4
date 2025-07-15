import json
import sys
from pathlib import Path
import fakeredis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config as cfg_mod

def test_load_and_save(tmp_path):
    conf_path = tmp_path / "c.json"
    conf_path.write_text(json.dumps({"redis_url": "redis://localhost:6379/0"}))
    data = cfg_mod.load_config(str(conf_path))
    assert data["redis_url"] == "redis://localhost:6379/0"
    r = fakeredis.FakeRedis()
    cfg_mod.save_config(data, str(conf_path), r)
    stored = json.loads(r.get("config"))
    assert stored["redis_url"] == "redis://localhost:6379/0"
