import importlib
import os
from pathlib import Path
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import inspect
import pkgutil
from typing import Dict, Any, List, Type
from .core import IMetric, Evaluator, ScoreAggregator # Ensure these are imported from core
BASE_PATH = Path(__file__).resolve().parent
DEV_MODE = os.getenv("LONO_MODE", "prod").lower() in ("dev", "debug")
_loaded_modules: Dict[str, Any] = {}
_registered_metrics: Dict[str, Any] = {}
logging.basicConfig(
    level=logging.DEBUG if DEV_MODE else logging.INFO,
    format='[LONO_LIBS] %(levelname)s: %(message)s'
)
_logger = logging.getLogger(__name__)
class _Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
def _log_dev(message: str, level: str = "info"):
    if not DEV_MODE:
        return
    color = {
        "info": _Color.CYAN,
        "ok": _Color.GREEN,
        "warn": _Color.YELLOW,
        "err": _Color.RED
    }.get(level, _Color.CYAN)
    print(f"{color}[LONO_LIBS] {message}{_Color.RESET}")
def _discover_subfolders(base_path: Path):
    return [f.name for f in base_path.iterdir() if f.is_dir() and f.name not in ['core', 'tests', 'docs', 'examples', '__pycache__']]
def _gather_module_paths_for_loading(base_path: Path):
    paths = []
    for pyfile in base_path.glob("*.py"):
        if pyfile.stem not in ["__init__", "best_result", "summary_reports", "unified_runner", "data_prep", "models", "reporting", "visualization"]:
            paths.append(pyfile.stem)
    subfolders_to_scan = _discover_subfolders(base_path)
    for sub in subfolders_to_scan:
        subdir = base_path / sub
        for pyfile in subdir.glob("*.py"):
            if pyfile.stem != "__init__":
                paths.append(f"{sub}.{pyfile.stem}")
    core_modules = ['evaluator', 'scoring']
    for core_mod in core_modules:
        paths.append(f"core.{core_mod}")
    return paths
def _load_single_module_and_store(module_path: str):
    start_time = time.perf_counter()
    try:
        mod = importlib.import_module(f"lono_libs.{module_path}")
        duration = (time.perf_counter() - start_time) * 1000
        _log_dev(f"Loaded module: {module_path} ({duration:.2f} ms)", "ok")
        return module_path, mod
    except ModuleNotFoundError:
        duration = (time.perf_counter() - start_time) * 1000
        _logger.debug(f"Module not found: {module_path} ({duration:.2f} ms)")
        _log_dev(f"Module not found: {module_path} ({duration:.2f} ms)", "warn")
    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        _logger.error(f"Error importing '{module_path}': {e}", exc_info=DEV_MODE)
        _log_dev(f"Error importing '{module_path}': {e} ({duration:.2f} ms)", "err")
    return None, None
_start_time_init = time.perf_counter()
_log_dev("LONO_LIBS initialization started.", "info")
_modules_to_load_paths = _gather_module_paths_for_loading(BASE_PATH)
if _modules_to_load_paths:
    if DEV_MODE:
        _log_dev("Starting parallel module loading...", "info")
    max_workers = min(len(_modules_to_load_paths) if _modules_to_load_paths else 1, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        load_results = list(executor.map(_load_single_module_and_store, _modules_to_load_paths))
    for path, mod in load_results:
        if path and mod:
            _loaded_modules[path] = mod
else:
    _log_dev("No modules found to load via parallel discovery.", "warn")
_public_api_elements = []
try:
    from .data_prep import create_polynomial_features, apply_standard_scaler, preprocess_data, get_preprocessing_pipeline
    globals()['create_polynomial_features'] = create_polynomial_features
    globals()['apply_standard_scaler'] = apply_standard_scaler
    globals()['preprocess_data'] = preprocess_data
    globals()['get_preprocessing_pipeline'] = get_preprocessing_pipeline
    _public_api_elements.extend(['create_polynomial_features', 'apply_standard_scaler', 'preprocess_data', 'get_preprocessing_pipeline'])
except ImportError as e: _logger.debug(f"Could not expose data_prep functions: {e}")
try:
    from .models import get_classification_models, get_regression_models, create_model_instance
    globals()['get_classification_models'] = get_classification_models
    globals()['get_regression_models'] = get_regression_models
    globals()['create_model_instance'] = create_model_instance
    _public_api_elements.extend(['get_classification_models', 'get_regression_models', 'create_model_instance'])
except ImportError as e: _logger.debug(f"Could not expose models functions/classes: {e}")
try:
    from .reporting import ReportingGenerator
    globals()['ReportingGenerator'] = ReportingGenerator
    _public_api_elements.append('ReportingGenerator')
except ImportError as e: _logger.debug(f"Could not expose ReportingGenerator: {e}")
try:
    from .core.evaluator import Evaluator
    globals()['Evaluator'] = Evaluator
    _public_api_elements.append('Evaluator')
except ImportError as e: _logger.debug(f"Could not expose Evaluator: {e}")
try:
    from .core.scoring import ScoreAggregator
    globals()['ScoreAggregator'] = ScoreAggregator
    _public_api_elements.append('ScoreAggregator')
except ImportError as e: _logger.debug(f"Could not expose ScoreAggregator: {e}")
try:
    from .best_result import BestResultFinder
    globals()['BestResultFinder'] = BestResultFinder
    _public_api_elements.append('BestResultFinder')
except ImportError as e: _logger.debug(f"Could not expose BestResultFinder: {e}")
try:
    from .summary_reports import SummaryReportGenerator
    globals()['SummaryReportGenerator'] = SummaryReportGenerator
    _public_api_elements.append('SummaryReportGenerator')
except ImportError as e: _logger.debug(f"Could not expose SummaryReportGenerator: {e}")
try:
    from .unified_runner import UnifiedRunner
    globals()['UnifiedRunner'] = UnifiedRunner
    _public_api_elements.append('UnifiedRunner')
except ImportError as e: _logger.debug(f"Could not expose UnifiedRunner: {e}")
try:
    from .visualization import VisualizationGenerator
    globals()['VisualizationGenerator'] = VisualizationGenerator
    _public_api_elements.append('VisualizationGenerator')
except ImportError as e: _logger.debug(f"Could not expose VisualizationGenerator: {e}")
_metric_packages_for_discovery = ['classification', 'regression']
_log_dev(f"Discovering and registering metric classes...", "info")
for package_name in _metric_packages_for_discovery:
    for _module_path, _module_obj in _loaded_modules.items():
        if _module_path.startswith(f"{package_name}."):
            for _member_name, _member_obj in inspect.getmembers(_module_obj, inspect.isclass):
                if isinstance(_member_obj, type) and hasattr(_member_obj, 'calculate') and hasattr(_member_obj, 'name') and _member_obj is not IMetric:
                    _registered_metrics[_member_obj.name] = _member_obj
                    globals()[_member_obj.name] = _member_obj
                    _public_api_elements.append(_member_obj.name)
                    _log_dev(f"Registered metric class: {_member_obj.name} from {_module_path}", "ok")
__all__ = sorted(list(set(_public_api_elements)))
_end_time_init = time.perf_counter()
_log_dev(f"LONO_LIBS initialization completed in {(_end_time_init - _start_time_init)*1000:.2f} ms", "info")
if DEV_MODE:
    _log_dev(f"Loaded {len(__all__)} public API elements successfully.", "info")
    _log_dev("Hybrid mode: Development logging and performance insights are enabled 🧠", "info")

def get_all_metrics() -> Dict[str, Any]:
    return _registered_metrics

