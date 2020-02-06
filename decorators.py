import time
import numpy as np

# decorator for ignoring divide warnings 
def ignore_div_warn(func): 

    def div_wrapper(*args, **kwargs): 
        
        old_err_setting = np.geterr();
        np.seterr(divide='ignore', invalid='ignore'); 
  
        func_return = func(*args, **kwargs) 
  
        np.seterr(**old_err_setting);
        
        return func_return
  
    return div_wrapper 


# decorator for printing elapsed time 
def time_it(func): 

    def timer_wrapper(*args, **kwargs): 
        
        start = time.time() 
  
        func_return = func(*args, **kwargs) 
  
        stop = time.time()
    
        print(f'elapsed time: {(stop - start):.2f} s');
        
        return func_return
  
    return timer_wrapper 
