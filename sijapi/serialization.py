# serialization.py

import json
from typing import Any
from uuid import UUID
from decimal import Decimal
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from datetime import datetime as dt_datetime, date, time
from .logs import get_logger

l = get_logger(__name__)

def serialize(obj: Any) -> Any:
    """Serializer for database inputs that keeps datetime objects intact"""
    if isinstance(obj, (dt_datetime, date, time)):
        return obj
    return json_serial(obj)


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (dt_datetime, date)):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [json_serial(item) for item in obj]
    if isinstance(obj, dict):
        return {json_serial(k): json_serial(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [json_serial(item) for item in obj]
    if isinstance(obj, tuple):
        return list(json_serial(item) for item in obj)
    if isinstance(obj, np.ndarray):
        return json_serial(obj.tolist())
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return json_serial(obj.to_dict())
    if obj is None:
        return None
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, range):
        return {'start': obj.start, 'stop': obj.stop, 'step': obj.step}
    if hasattr(obj, '__iter__'):
        return list(json_serial(item) for item in obj)
    if hasattr(obj, '__dict__'):
        return {k: json_serial(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    raise TypeError(f"Type {type(obj)} not serializable")


def json_dumps(obj: Any) -> str:
    """
    Serialize obj to a JSON formatted str using the custom serializer.
    """
    return json.dumps(obj, default=json_serial)

def json_loads(json_str: str) -> Any:
    """
    Deserialize json_str to a Python object.
    """
    return json.loads(json_str)
