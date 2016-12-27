def decorator_args(decorator):
    def decorator_creator(*args, **kwargs):
        def decorator_wrapper(func):
            return decorator(func, *args, **kwargs)
        return decorator_wrapper        
    return decorator_creator

@decorator_args
def test_func2(func, val):
    def wrapper(*args, **kwargs):
        print('test1', val)
        func(*args, **kwargs)
        print('test2', val)
    return wrapper

@test_func2(5)
def real_func(a, s):
    print('Real func {0}, {1}'.format(a, s))


real_func(3, 'f')