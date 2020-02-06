import numpy as np

from decorators import *

@time_it
@ignore_div_warn
def estimate_alb_norm(image_stack, scriptV, shadow_trick=False):
    
    h, w, n, n_ch = image_stack.shape
    
    if shadow_trick:
        
        print('computing albedo and normals with shadow trick...', end='')
        
        albedo = np.empty((h, w, n_ch))
        normals = np.empty((h, w, 3, n_ch))
        res_map = np.empty((h, w, n_ch))
        
        for x in range(h):
            for y in range(w):
                for ch in range(n_ch):  
                    i_x_y = image_stack[x, y, :, ch]
                    scriptI = np.diag(i_x_y)
                    g_x_y, res, _, _ = np.linalg.lstsq(scriptI @ scriptV, scriptI @ i_x_y, rcond=None)
            
                    g_norm = np.linalg.norm(g_x_y)
                
                    albedo[x, y, ch] = g_norm
                    normals[x, y, :, ch] = g_x_y.squeeze() / g_norm 
                
                    res_map[x, y, ch] = res if res.shape == (1,) else 0
    
    else:
        
        print('computing albedo and normals without shadow trick...', end='')
        
        i_stack = image_stack.swapaxes(2, 3).reshape((-1, n)).T
        g_stack, res, rank, s = np.linalg.lstsq(scriptV, i_stack, rcond=None)

        g_norm = np.linalg.norm(g_stack, axis=0)
        albedo = g_norm.reshape(h, w, -1)
        
        normals = g_stack / g_norm
        normals = normals.T.reshape((h, w, -1, 3)).swapaxes(2, 3)
        
        res_map = res.T.reshape(h, w, -1)

    #np.clip(albedo, 0, 1, albedo)
    print('finished!')
    
    return albedo, normals, res_map


@time_it
@ignore_div_warn
def check_integrability(normals, threshold=0.005, nan_to_zero=False):
    N1 = normals[:,:, 0]
    N2 = normals[:,:, 1]
    N3 = normals[:,:, 2]

    p = N1/N3
    q = N2/N3

    if nan_to_zero:
        p = np.nan_to_num(p)
        q = np.nan_to_num(q)

    dp_d2 = np.gradient(p, axis=(0, 1))[0]
    dq_d1 = np.gradient(q, axis=(0, 1))[1]
    SE = (dp_d2 - dq_d1)**2
    
    print('threshold: ', threshold)
    print('-'*40)
    h, w, n_ch = SE.shape
    for i in range(n_ch):
        n_outliers = np.sum(SE[:,:,i] > threshold)
        perc_outliers = 100 * n_outliers / (w*h)
        print(f'number of outliers in channel {i+1}: ', n_outliers)
        print(f'percentage of outliers in channel {i+1}: ', f'{perc_outliers:0.1f}%')
        print('-'*40)
    
    return p, q, SE


@time_it
def construct_surface(p, q, path_type='column'):
    
    path_types = {'column', 'row', 'average'}
    
    if path_type not in path_types:
        raise ValueError(f'unknow path_type: \'{path_type}\'')
    
    if path_type == 'column':
        print('integration in column-major order...')
        a = np.nan_to_num(q[:, 0, :])
        height_map = np.nan_to_num(p)
        a[0, :] = 0
        height_map[:, 0, :] = np.cumsum(a, axis=0)
        height_map = np.cumsum(height_map, axis=1)
    
    if path_type == 'row':
        print('integration in row-major order...')
        a = np.nan_to_num(p[0, :, :])
        height_map = np.nan_to_num(q)
        a[0, :] = 0
        height_map[0, :, :] = np.cumsum(a, axis=1)
        height_map = np.cumsum(height_map, axis=0)  
    
    if path_type == 'average':
        height_map_col = construct_surface(p, q, path_type='column')
        height_map_row = construct_surface(p, q, path_type='row')
        print('averaging...')
        height_map = (height_map_col + height_map_row) / 2.0
    
    return height_map