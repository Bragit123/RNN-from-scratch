class A:
    def __init__(self):
        if type(self) == B:
            print("Yes!")
        else:
            print("No!")

class B(A):
    def __init__(self):
        super().__init__()

first = A()
second = B()