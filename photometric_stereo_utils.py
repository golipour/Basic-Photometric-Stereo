from pathlib import Path
import PIL
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from decorators import * 


def normalize(x):
    
    x_tmp = x[~np.isnan(x)]
    
    if x_tmp.size != 0:
        x_min = np.min(x_tmp)
        x_max = np.max(x_tmp)
        x_ = (x - x_min) / (x_max - x_min)
        x_[np.isnan(x)] = np.nan
    else:
        x_ = np.array(x)
        
    return x_


def read_V(im_path):
    
    im_name = im_path.name.rstrip('.png')
    src_vec_str = im_name.rsplit(sep='_', maxsplit=3)[1:]
    src_vec = [float(x) for x in src_vec_str]
    src_vec.append(0.5)
    src_vec[0] = - src_vec[0]
    #src_vec[0] = - src_vec[0] if (src_vec[0] != 0) else 0
    S = np.array(src_vec)
    V = S / np.linalg.norm(S)
    
    return V


def read_face_V(im_path):
    
    im_name = im_path.name.rstrip('.png')
    i = im_name.find('A') + 1
    j = im_name.find('E') + 1
    theta = float(im_name[i:i+4]) * (np.pi/180.0)
    phi = float(im_name[j:j+3]) * (np.pi/180.0)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    V = np.array([y, z, x])
    
    return V


@time_it
def load_syn_images(image_dir, channels=(0, 1, 2)):
    
    p = Path(image_dir)
    img_path = list(p.glob('*.png'))
    
    if len(img_path)==0: 
        raise FileNotFoundError('the image directory is empty.')
    
    for idx, img_file in enumerate(img_path):
    
        with PIL.Image.open(img_file) as img:
            if idx == 0:
                w, h = img.size
                n = len(img_path)
                n_ch = len(channels)
                image_stack = np.empty((h, w, n, n_ch))
                scriptV = np.empty((n, 3))
                print('loading images...', end='')
    
            scriptV[idx, :] = read_V(img_file)
    
            for j in range(len(channels)):
                ch = channels[j]
                image_stack[:, :, idx, j] = np.asarray(img.getchannel(ch), dtype=np.float) / 255.0
    
    print('finished!')    
    print('number of images:  ', image_stack.shape[2])
    print('image size:        ', image_stack.shape[0:2])
    print('number of channels:', image_stack.shape[3])
    
    return image_stack, scriptV


@time_it
def load_face_images(image_dir, channels=(0,)):
    
    p = Path(image_dir)
    
    ambient_img_path = list(p.glob('yaleB02_P00_Ambient.pgm'))[0]
    ambient_img = np.asarray(PIL.Image.open(ambient_img_path), dtype=np.float)
    
    img_path = list(p.glob('yaleB02_P00A*.pgm'))

    for idx, img_file in enumerate(img_path):
    
        with PIL.Image.open(img_file) as img:
            
            if idx == 0:
                w, h = img.size
                n = len(img_path)
                n_ch = len(channels)
                image_stack = np.empty((h, w, n, n_ch), dtype=np.float)
                scriptV = np.empty((n, 3), np.float)
                #ambient_img = np.asarray(ambient_img).reshape(h, w, 1, n_ch)
                print('loading images...', end='')
    
            scriptV[idx, :] = read_face_V(img_file)
    
            for j in range(len(channels)):
                ch = channels[j]
                tmp_img = np.asarray(img.getchannel(ch), dtype=np.float) 
                image_stack[:, :, idx, j] = (tmp_img - ambient_img) / 255.0
            
            #image_stack.__iadd__(-ambient_img)
            #image_stack.__imul__(1.0/ 255.0)
            #image_stack = normalize(image_stack)
            np.clip(image_stack, 0, None, out=image_stack)
    
    print('finished!')    
    print('number of images:  ', image_stack.shape[2])
    print('image size:        ', image_stack.shape[0:2])
    print('number of channels:', image_stack.shape[3])
    
    return image_stack, scriptV


def show_results(albedo, normals, fig_size=3, nan_to_zero=True):
    
    n_ch = albedo.shape[2]
    
    normals_ = normalize(normals)
    N = np.empty(normals.shape)

    for i in range(3):
        N[:,:,i] = normalize(normals[:,:,i])

    normals_ = np.nan_to_num(normals_)    
    N = np.nan_to_num(N)
    
    fig = plt.figure(figsize=(fig_size*5, fig_size*n_ch), tight_layout=True)

    imgs = [albedo, normals_, N[:,:,0], N[:,:,1], N[:,:,2]]
    
    titles = ['albedo', 'normal map', 'normal x-component', 
              'normal y-component', 'normal z-component']

    for j in range(5):
        for i in range(n_ch):
            ax = fig.add_subplot(n_ch, 5, 5*i+j+1)
            im = imgs[j].T[i].T
            ax.imshow(im, cmap='gray')
            ax.set(xticks=[], yticks=[])
            if j==0: ax.set_ylabel(f'channel {i+1}', fontsize=12)
            if i==0: 
                ax.set_xlabel(titles[j], fontsize=12)
                ax.xaxis.set_label_position('top')


def show_model(height_map, albedo, fig_size=5, elev=25, azim=-60):
    
    h, w, n_ch = height_map.shape
    
    fig = plt.figure(figsize=(fig_size*n_ch, fig_size), tight_layout=True)
    
    X = np.arange(w)
    Y = np.arange(h)
    X, Y = np.meshgrid(X, Y)

    l = max(w, h)/2.0 
    
    for i in range(n_ch):
        ax = fig.add_subplot(1, n_ch, i+1, projection='3d')
        Z = height_map[:, :, i]
        A = albedo[:, :, i]
    
        ax.plot_surface(X, Z, Y, facecolors=cm.gray(A))
    
        ax.set(xlim=(0, w), ylim=(l, -l), zlim=(h, 0))
        ax.set(xlabel='x', ylabel='z', zlabel='y')
        ax.set(title = f'model for channel {i+1}')
        ax.view_init(elev, azim)


def show_normals(normals, height_map, sampling_step=8, fig_size=5, elev=50, azim=60):
    
    h, w, n_ch = height_map.shape
    
    #sub-sampling step size
    s = sampling_step
    #s = min(w, h)//n_samples
    
    h_new, w_new = height_map[::s,::s, 0].shape[0:2]
    X, Y = np.meshgrid(np.arange(w_new), 
                       np.arange(h_new));
    
    fig = plt.figure(figsize=(fig_size*n_ch, fig_size), tight_layout=True)    

    for i in range(n_ch):
    
        ax = fig.add_subplot(1, n_ch, i+1, projection='3d')
        H = np.nan_to_num(height_map[::s,::s, i])

        N1 = np.nan_to_num(normals[::s, ::s, 0, i])
        N2 = np.nan_to_num(normals[::s, ::s, 1, i])
        N3 = np.nan_to_num(normals[::s, ::s, 2, i])
    
        ax.quiver(X, Y, H, -N1, -N2, N3, length=2, arrow_length_ratio=.4)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        #ax.set_zbound(lower=0, upper=50)
        ax.set_zlim(-max(h, w)/2, max(h, w)/2)
        
        ax.view_init(elev, azim)


def show_samples(image_stack, scriptV, n_samples=5, fig_size=3, show_channels=False):

    _, _, n, n_ch = image_stack.shape

    sample = np.random.choice(n, size=n_samples, replace=False)

    if n_ch>1 and show_channels:
        figsize=(fig_size*n_samples, fig_size*(n_ch+1))
        fig, axes = plt.subplots(n_ch+1, n_samples, figsize=figsize, tight_layout=True)
        axes = axes.reshape(n_ch+1, n_samples)
    else:
        figsize=(fig_size*n_samples, fig_size)
        fig, axes = plt.subplots(1, n_samples, figsize=figsize, tight_layout=True)
        axes = axes.reshape(1, n_samples)

    for j in range(n_samples):
        ax = axes[0, j]
        im=image_stack[:, :, sample[j], :]
        if im.shape[2] not in [0, 3]:
            im = np.mean(im, axis=-1)
        im = im.squeeze()
        title = 'V='+np.array2string(scriptV[sample[j], :], precision=2, separator=', ')
        ax.imshow(im, cmap='gray')
        ax.set(xticks=[], yticks=[]);
        ax.set_title(title, fontsize=12)
        if j==0: ax.set_ylabel('all channels', fontsize=12)

    if n_ch>1 and show_channels:
        color = ['Reds', 'Greens', 'Blues']
        for j in range(n_samples):
            for i in range(n_ch):    
                ax = axes[i+1, j]
                ax.imshow(image_stack[:, :, sample[j], i], cmap=color[i])
                ax.set(xticks=[], yticks=[]);
                if j==0: ax.set_ylabel(f'channel {i+1}', fontsize=12)


def show_samples_per_row(image_stack, scriptV, n_samples=5, fig_size=2.5, show_channels=True):
    
    _, _, n, n_ch = image_stack.shape

    sample = np.random.choice(n, size=n_samples, replace=False)

    if n_ch>1 and show_channels:
        figsize=(fig_size*(n_ch+1), fig_size*n_samples)
        fig, axes = plt.subplots(n_samples, n_ch+1, figsize=figsize, tight_layout=True)
        axes = axes.reshape(n_samples, n_ch+1)
    else:
        figsize=(fig_size, fig_size*n_samples)
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize, tight_layout=True)
        axes = axes.reshape(n_samples, 1)

    for j in range(n_samples):
        ax = axes[j, 0]
        im=image_stack[:, :, sample[j], :].squeeze()
        if im.shape[2] not in [0, 3]:
            im = np.mean(im, axis=-1)
        title = 'V='+np.array2string(scriptV[sample[j], :], precision=1, separator=', ')
        ax.imshow(im, cmap='gray')
        ax.set(xticks=[], yticks=[]);
        if j==0: ax.set_title('all channels', fontsize=11)
        ax.set_ylabel(title, fontsize=10)

    if n_ch>1 and show_channels:
        color = ['Reds', 'Greens', 'Blues']
        for j in range(n_samples):
            for i in range(n_ch):    
                ax = axes[j, i+1]
                ax.imshow(image_stack[:, :, sample[j], i], cmap=color[i])
                ax.set(xticks=[], yticks=[]);
                if j==0: ax.set_title(f'channel {i+1}', fontsize=11)


def show_outlaiers(SE, elev=60, azim=70):
    
    h, w , n_ch= SE.shape
    X, Y = np.meshgrid(np.arange(w), 
                       np.arange(h));
    
    fig = plt.figure(figsize=(5*n_ch, 5))

    for i in range(n_ch):
        ax = fig.add_subplot(1, n_ch, i+1, projection='3d')
        ax.plot_surface(X, Y, SE[:, :, i], cmap='Reds')
        ax.set_zlim(0.0, SE.max());
        ax.set_title(f'outlaiers in channel {i+1}', fontsize=12);
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ax.invert_xaxis()
        ax.view_init(60, 70)
                    
