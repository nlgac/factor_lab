"""
Analysis Framework for Factor Models
=====================================

Provides the core infrastructure for custom simulation analyses.

The framework consists of:
- SimulationAnalysis: Protocol defining the analysis interface
- SimulationContext: Immutable snapshot of simulation state

Examples
--------
>>> from factor_lab.analysis import SimulationAnalysis, SimulationContext
>>> 
>>> class MyAnalysis(SimulationAnalysis):
...     def analyze(self, context: SimulationContext):
...         return {"metric": compute(context.model)}
"""

from .protocol import SimulationAnalysis
from .context import SimulationContext

__all__ = ['SimulationAnalysis', 'SimulationContext']
