# import functools
# from typing import Callable, Optional
# from injector import inject, Injector
# from .events import State


# def every_epochs(func: Callable, n: int = 1):

#     @inject
#     def wrapper(state: State, **kwargs):
#         if state.every_epochs(n):
#             # return container.call_with_injection(
#             #     func,
#             #     args=args,
#             #     kwargs=kwargs,
#             # )
#             return func(**kwargs)

#     original_annotations = func.__annotations__
#     wrapper.__annotations__.update(original_annotations)

#     return wrapper


# def every_iterations(func: Optional[Callable] = None, n: int = 1):

#     # @functools.wraps(func)
#     def wrapper(state: State = None, **kwargs):
#         if state.every_iterations(n):
#             func(**kwargs)

#     original_annotations = func.__annotations__

#     print(original_annotations)
    
#     wrapper.__annotations__ = 

#     wrapper = inject(wrapper)


#     return wrapper
