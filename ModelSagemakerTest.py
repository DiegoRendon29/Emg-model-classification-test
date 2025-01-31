import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.clarify import BiasConfig, ModelMonitor, DataConfig, ExplainabilityConfig
from sagemaker.clarify import ClarifyProcessor
from sagemaker import get_execution_role

role = get_execution_role()