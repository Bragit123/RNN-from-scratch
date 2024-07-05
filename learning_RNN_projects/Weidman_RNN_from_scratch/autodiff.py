
from typing import Union, List
import numpy as np

a = np.array([3,3])
print("Addition using '__add__':", a.__add__(4))
print("Addition using '+':", a+4)

Numerable = Union[float, int]

# Declare NumberWithGrad to avoid problems with type hints
class NumberWithGrad:
    pass

def ensure_number(num: Numerable) -> NumberWithGrad:
    if isinstance(num, NumberWithGrad):
        return num
    else:
        return NumberWithGrad(num)
    

class NumberWithGrad():
    def __init__(self,
                 num: Numerable,
                 depends_on: List[Numerable] = None,
                 creation_op: str = ''):
        self.num = num
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op
    
    def __add__(self, other: Numerable) -> NumberWithGrad:
        return NumberWithGrad(self.num + ensure_number(other).num,
                              depends_on = [self, ensure_number(other)],
                              creation_op = 'add')
    
    def __mul__(self,
                other: Numerable = None) -> NumberWithGrad:
         return NumberWithGrad(self.num * ensure_number(other).num,
                               depends_on = [self, ensure_number(other)],
                               creation_op = 'mul')
    
    def backward(self, backward_grad: Numerable = None) -> None:
        if backward_grad is None: # First time calling backward
            self.grad = 1
        else:
            # These lines allow gradients to accumulate.
            # If the gradient doesn't exist yet, simply set it equal
            # to backward_grad.
            if self.grad is None:
                self.grad = backward_grad
            # Otherwise, simply add backward_grad to the existing gradient
            else:
                self.grad += backward_grad
        
        if self.creation_op == "add":
            # Simply send backward self.grad, since increasing either of these
            # elements will increase the output by that same amount
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)
        
        if self.creation_op == "mul":
            # Calculate the derivative with respect to the first element
            new = self.depends_on[1] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[0].backward(new.num)

            # Calculate the derivative with respect to the second element
            new = self.depends_on[0] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[1].backward(new.num)