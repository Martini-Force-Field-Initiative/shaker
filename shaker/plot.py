import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.image as mpimg
mpl.rcParams['figure.dpi'] = 150
import matplotlib.patches as mpatches
import os
import subprocess

'''
Collection of functions to assist in plotting and imaging.
'''

def calculate_non_transparent_bounds(image):
    """
    Calculate the bounding box of the non-transparent region of an image.
    
    Parameters:
        image (numpy.ndarray): The image array (with an alpha channel for transparency).
        
    Returns:
        tuple: (x_min, x_max, y_min, y_max) bounds of the non-transparent region.
    """
    if image.shape[-1] == 4:  # Check if the image has an alpha channel (RGBA)
        alpha_channel = image[..., 3]
        non_transparent_indices = np.where(alpha_channel > 0)
        y_min, y_max = non_transparent_indices[0].min(), non_transparent_indices[0].max()
        x_min, x_max = non_transparent_indices[1].min(), non_transparent_indices[1].max()
        return x_min, x_max, y_min, y_max
    else:
        raise ValueError("The image does not have an alpha channel (RGBA).")

def best_grid(n: int):
    """Return (nrows, ncols) for n plots using a near-square layout."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def predict_figsize(grids, cell_w=3.2, cell_h=1.5, section_gap_h=0.6,
                    left=0.8, right=0.2, top=0.6, bottom=0.6):
    """
    grids: list of (nrows, ncols) for sections stacked vertically
    returns (fig_w, fig_h) in inches
    """
    max_cols = max(c for r, c in grids)
    total_rows = sum(r for r, c in grids)

    fig_w = left + max_cols * cell_w + right
    fig_h = bottom + total_rows * cell_h + top + section_gap_h * (len(grids) - 1)
    return fig_w, fig_h

def read_SASA_xvg(xvg):
    '''
    Reader for the SASA per residue .xvg file. Retrieves AVG and Std.
    '''
    rows = [l.split() for l in Path(xvg).read_text().splitlines()
            if l and l[0] not in "#@"]
    a = np.array(rows, float)
    return a[0, 1], a[0, 2]   

def plot_sasa_dir(root="./SASA",
                  xvg="resarea_SASA.xvg"):
    '''
    Scrapes the SASA directory and plots the SASA values for comparison.
    Returns fig, ax, items.
    '''
    root = Path(root)
    items = [(d.name, *read_SASA_xvg(d / xvg))
             for d in sorted(root.iterdir())
             if d.is_dir() and (d / xvg).exists()]

    names, vals, errs = zip(*items)

    x = np.arange(len(names))
    fig, ax = plt.subplots(
        figsize=(max(6, 0.8*len(names)), 4),
        tight_layout=True)

    ax.bar(x, vals, yerr=errs, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("SASA (nm$^2$)", fontweight='bold')
    fig.savefig("SASABar.png", dpi=300, transparent=True, bbox_inches='tight')

    return fig, ax, items


def render_Mapping(cg_mapped_gro, cg_mapped_xtc,
                   aa_gro, aa_xtc,
                   vmd='vmd', 
                   vmd_resolution=1000, 
                   render=True,
                   mogrify=True,
                   plot=True):
    '''
    Wrapper to write the vmd visualization states to visualize the CG Mapping overlayed
    on top of the AA structure .
    Also renders automatically and plots the renders.
    '''
    
    vmdvis = files("shaker.data.vmd_visualization") / "Mapping_render.vmd"
    
    with open(vmdvis) as topinput:
        top = topinput.readlines()
    top[351]=f'mol new {cg_mapped_gro} type gro first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[352]=f'mol addfile {cg_mapped_xtc} type xtc first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[392]=f'mol new {aa_gro} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[393]=f'mol addfile {aa_xtc} type xtc first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    with open('Mapping_render.vmd', 'w+') as topout:
        for line in top:
            topout.write(line)
    with open('Mapping.vmd', 'w+') as topout:
        for line in top[:662]: ## Excludes the lines that automagically render stuff.
            topout.write(line)

    if render:
        print('Rendering.....')
        cmd = [vmd,
               "-dispdev", "text",
               "-e", "Mapping_render.vmd",
               "-size", str(vmd_resolution), str(vmd_resolution),]
        logfile = "vmd_mapping_render.log"
        with open(logfile, "w") as log:
            subprocess.run(cmd,stdout=log, stderr=subprocess.STDOUT,check=True)
        print('.....Rendering Complete!')

    if mogrify:
        subprocess.call('mogrify -format png -transparent white *.tga'
                        , shell = True)    
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)

        ax = axs[0]
        img = mpimg.imread('./MAPzz.png')
        x_min, x_max, y_min, y_max = calculate_non_transparent_bounds(img)
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  # Note: y is inverted in image coordinates
        ax.set_axis_off()
        
        ax = axs[1]
        img = mpimg.imread('./MAPyy.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  
        ax.set_axis_off()    
        
        ax = axs[2]
        img = mpimg.imread('./MAPxx.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100) 
        ax.set_axis_off()

        fig.savefig("Mapping.png", dpi=300, transparent=True, bbox_inches='tight')


def render_connely_surface(vmd='vmd', 
                           vmd_resolution=1000, 
                           render=True,
                           mogrify=True,
                           plot=True):
    '''
    Wrapper to write the vmd visualization states to visualize the connely surfaces.
    Also renders automatically and plots the renders.
    '''
    
    vmdvis = files("shaker.data.vmd_visualization") / "surface.vmd"
    os.system(f'cp {vmdvis} .')
    vmdvis = files("shaker.data.vmd_visualization") / "surface_render.vmd"
    os.system(f'cp {vmdvis} .')

    if render:
        print('Rendering.....')
        cmd = [vmd,
               "-dispdev", "text",
               "-e", "surface_render.vmd",
               "-size", str(vmd_resolution), str(vmd_resolution),]
        logfile = "vmd_surface_render.log"
        with open(logfile, "w") as log:
            subprocess.run(cmd,stdout=log, stderr=subprocess.STDOUT,check=True)
        print('.....Rendering Complete!')

    if mogrify:
        subprocess.call('mogrify -format png -transparent white *.tga'
                        , shell = True)    
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)

        ax = axs[0]
        img = mpimg.imread('./SASAzz.png')
        x_min, x_max, y_min, y_max = calculate_non_transparent_bounds(img)
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  # Note: y is inverted in image coordinates
        ax.set_axis_off()
        
        ax = axs[1]
        img = mpimg.imread('./SASAyy.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  
        ax.set_axis_off()    
        
        ax = axs[2]
        img = mpimg.imread('./SASAxx.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100) 
        ax.set_axis_off()

        # Create fake legend handles
        blue_patch = mpatches.Patch(color='tab:blue', label='AA')
        red_patch  = mpatches.Patch(color='tab:red',  label='CG_Mapped')
        ax.legend(handles=[blue_patch, red_patch],
                  loc='upper right',     
                  frameon=False)

        fig.savefig("ConnelySurface.png", dpi=300, transparent=True, bbox_inches='tight')






        