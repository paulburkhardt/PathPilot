---
applyTo: '**'
---


Please use typehints and docstrings for the code you write.
There is no need for comments within the code.
The docstrings are supposed to be in the following format:

class Example:
    """
    Example class for showing how I want the docstrings to look like.
    
    Args:
        some_param: Some parameter to be passed here.
    Returns:
        -
    Raises:
        -
    """

    def __init__(
        self,
        some_param: int
    )->None:
        pass

    def some_method(
        self,
        some_arg: str
    )-> int:
        """
        Some method doing something.

        Args:
            some_arg: Some argument to be passed here.
        Returns:
            a fixed number 42.
        Raises:
            -
        """

        return 42