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


def plot_sasa_dir(root="./SASA",
                  xvg="resarea_SASA.xvg"):
    '''
    Plot SASA values from multiple simulations stored in subdirectories.

    This function scans the specified directory for subdirectories containing
    a SASA results file (e.g. `resarea_SASA.xvg`). Each subdirectory is assumed
    to represent a different model or simulation condition. The SASA values
    and associated errors are extracted and displayed as a bar plot for
    comparison.

    Parameters
    ----------
    root : str or Path, optional
        Directory containing subdirectories with SASA output files.
        Default is "./SASA".
    xvg : str, optional
        Name of the GROMACS `.xvg` file containing SASA values within each
        subdirectory. Default is "resarea_SASA.xvg".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The axes containing the bar plot.
    items : list of tuple
        List containing the parsed SASA data in the form
        `(name, value, error)` for each subdirectory.

    Notes
    -----
    The function expects each subdirectory inside `root` to contain the
    specified `.xvg` file. The label used in the bar plot corresponds to
    the name of the subdirectory.

    The resulting figure is saved as `SASABar.png`.
    '''
    root = Path(root)
    items = [(d.name, *_read_SASA_xvg(d / xvg))
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
                   dir_out='.', 
                   render=True,
                   mogrify=True,
                   plot=True):
    '''
    Wrapper to write the vmd visualization states to visualize the CG Mapping overlayed
    on top of the AA structure .
    Also renders automatically and plots the renders.
    '''
    
    vmdvis = files("shaker.data.vmd_visualization") / "Mapping_render.vmd"

    ## Fix pathing
    cg_mapped_gro = Path(cg_mapped_gro).resolve()
    cg_mapped_xtc = Path(cg_mapped_xtc).resolve()
    aa_gro = Path(aa_gro).resolve()
    aa_xtc = Path(aa_xtc).resolve()
    dir_out = Path(dir_out).resolve()

    ## Make sure formats are supported.
    _check_ext(cg_mapped_gro, {".gro", ".pdb"})
    _check_ext(aa_gro, {".gro", ".pdb"})
    _check_ext(cg_mapped_xtc, {".xtc", ".trr"})
    _check_ext(aa_xtc, {".xtc", ".trr"})

    ## Prepare render dir.
    dir_writing = f'{dir_out}/Renders/'
    os.makedirs(dir_writing, exist_ok=True)

    ## Fix vis
    with open(vmdvis) as topinput:
        top = topinput.readlines()
        
    ##Careful! These are hardcoded.
    top[533]=f'mol new {cg_mapped_gro} type {Path(cg_mapped_gro).suffix.lower().lstrip(".")} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[534]=f'mol addfile {cg_mapped_xtc} type {Path(cg_mapped_xtc).suffix.lower().lstrip(".")} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[574]=f'mol new {aa_gro} type {Path(aa_gro).suffix.lower().lstrip(".")} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    top[575]=f'mol addfile {aa_xtc} type {Path(aa_xtc).suffix.lower().lstrip(".")} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n'
    
    with open(f'{dir_writing}/Mapping_render.vmd', 'w+') as topout:
        for line in top:
            topout.write(line)
    with open(f'{dir_writing}/Mapping.vmd', 'w+') as topout:
        for line in top[:845]: ## Excludes the lines that automagically render stuff.
            topout.write(line)
    ## render
    if render:
        print('Rendering.....')
        cmd = [vmd,
               "-dispdev", "text", "-e", "Mapping_render.vmd",
               "-size", str(vmd_resolution), str(vmd_resolution),]
        logfile = f'{dir_writing}/vmd_mapping_render.log'
        with open(logfile, "w") as log:
            subprocess.run(cmd,stdout=log, stderr=subprocess.STDOUT, check=True, cwd=dir_writing)
        print('.....Rendering Complete!')

    if mogrify:
        subprocess.call(["mogrify", "-format", "png", "-transparent", "white", "*.tga"]
                        , check=True, cwd=dir_writing)    
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)

        ax = axs[0]
        img = mpimg.imread(f'{dir_writing}/MAPzz.png')
        x_min, x_max, y_min, y_max = _calculate_non_transparent_bounds(img)
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  # Note: y is inverted in image coordinates
        ax.set_axis_off()
        
        ax = axs[1]
        img = mpimg.imread(f'{dir_writing}/MAPyy.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  
        ax.set_axis_off()    
        
        ax = axs[2]
        img = mpimg.imread(f'{dir_writing}/MAPxx.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100) 
        ax.set_axis_off()

        fig.savefig(f'{dir_writing}/Mapping.png', dpi=300, transparent=True, bbox_inches='tight')


def render_connely_surface(vmd='vmd', 
                           vmd_resolution=1000, 
                           dir_out='.', 
                           render=True,
                           mogrify=True,
                           plot=True):
    '''
    Wrapper to write the vmd visualization states to visualize the connely surfaces.
    Also renders automatically and plots the renders.
    '''

    ## Fix pathing
    dir_out = Path(dir_out).resolve()
    
    ## Prepare render dir.
    dir_writing = f'{dir_out}/Renders/'
    os.makedirs(dir_writing, exist_ok=True)

    ## Prepare vis
    vmdvis = files("shaker.data.vmd_visualization") / "surface.vmd"
    os.system(f'cp {vmdvis} {dir_writing}')
    vmdvis = files("shaker.data.vmd_visualization") / "surface_render.vmd"
    os.system(f'cp {vmdvis} {dir_writing}')

    ## render
    if render:
        print('Rendering.....')
        cmd = [vmd,
               "-dispdev", "text",
               "-e", "surface_render.vmd",
               "-size", str(vmd_resolution), str(vmd_resolution),]
        logfile = f"{dir_writing}/vmd_surface_render.log"
        with open(logfile, "w") as log:
            subprocess.run(cmd,stdout=log, stderr=subprocess.STDOUT,check=True, cwd=dir_writing)
        print('.....Rendering Complete!')

    if mogrify:
        subprocess.call(["mogrify", "-format", "png", "-transparent", "white", "*.tga"]
                        , check=True, cwd=dir_writing)   
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)

        ax = axs[0]
        img = mpimg.imread(f'{dir_writing}/SASAzz.png')
        x_min, x_max, y_min, y_max = _calculate_non_transparent_bounds(img)
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  # Note: y is inverted in image coordinates
        ax.set_axis_off()
        
        ax = axs[1]
        img = mpimg.imread(f'{dir_writing}/SASAyy.png')
        imgplot = ax.imshow(img[:, :, :]) 
        ax.set_xlim(x_min-100, x_max+100)
        ax.set_ylim(y_max+100, y_min-100)  
        ax.set_axis_off()    
        
        ax = axs[2]
        img = mpimg.imread(f'{dir_writing}/SASAxx.png')
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

        fig.savefig(f'{dir_writing}/ConnelySurface.png', dpi=300, transparent=True, bbox_inches='tight')


def _calculate_non_transparent_bounds(image):
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


def _read_SASA_xvg(xvg):
    '''
    Reader for the SASA per residue .xvg file. Retrieves AVG and Std.
    '''
    rows = [l.split() for l in Path(xvg).read_text().splitlines()
            if l and l[0] not in "#@"]
    a = np.array(rows, float)
    return a[0, 1], a[0, 2]   

def _check_ext(file, allowed):
    if Path(file).suffix.lower() not in allowed:
        raise ValueError(f"{file} must have one of: {', '.join(allowed)}")

## these 2 will become internal funcs once I build the 
## distribution plotter function.
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
        