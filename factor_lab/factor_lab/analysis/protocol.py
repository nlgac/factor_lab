"""
protocol.py - Core Protocol for Custom Analyses
================================================

Defines the protocol that all custom simulation analyses must implement.
"""

from typing import Protocol, Dict, Any, runtime_checkable

__all__ = ['SimulationAnalysis']


@runtime_checkable
class SimulationAnalysis(Protocol):
    """
    Protocol for custom simulation analyses.
    
    All custom analyses must implement the analyze() method which takes
    a SimulationContext and returns a dictionary of results.
    
    The protocol approach allows for duck typing - any class with an
    analyze() method matching this signature can be used as an analysis,
    whether or not it explicitly inherits from this protocol.
    
    Examples
    --------
    >>> class MyAnalysis:
    ...     def analyze(self, context: SimulationContext) -> Dict[str, Any]:
    ...         return {"my_metric": compute_something(context)}
    >>> 
    >>> # Can be used directly
    >>> analysis = MyAnalysis()
    >>> results = analysis.analyze(context)
    
    See Also
    --------
    SimulationContext : The context object passed to analyze()
    """
    
    def analyze(self, context: 'SimulationContext') -> Dict[str, Any]:
        """
        Perform analysis on simulation context.
        
        Parameters
        ----------
        context : SimulationContext
            Immutable snapshot of simulation state containing:
            - model: The factor model
            - security_returns: Simulated returns
            - test_results: Statistical test results (if available)
            - And other simulation outputs
        
        Returns
        -------
        Dict[str, Any]
            Analysis results as key-value pairs.
            Keys should be descriptive metric names.
            Values can be any serializable Python object (floats, arrays, etc.).
        
        Examples
        --------
        >>> def analyze(self, context):
        ...     return {
        ...         "frobenius_error": np.linalg.norm(
        ...             context.model.B - estimated_B
        ...         ),
        ...         "correlation": compute_correlation(...),
        ...     }
        """
        ...
