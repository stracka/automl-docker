import logging
FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
_has_autogluon = importlib.util.find_spec("autogluon") is not None
_has_autosklearn = importlib.util.find_spec("autosklearn") is not None
_has_hepml = importlib.util.find_spec("hep_ml") is not None
_has_optuna = importlib.util.find_spec("optuna") is not None
_has_tpot = importlib.util.find_spec("tpot") is not None

if not _has_optuna: 
    logger.warning("Optuna not installed.")

if not _has_tpot:
    logger.warning("TPOT not installed.")

if not _has_hepml:
    logger.warning("hep_ml not installed.")
    
if not _has_autogluon:
    logger.warning("AutoGluon not installed.")

if not _has_autosklearn:
    logger.warning("AutoSklearn not installed.")
