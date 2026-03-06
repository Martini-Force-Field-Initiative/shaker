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
import nglview as nv
import MDAnalysis as md
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

def render_mapping(cg_mapped_gro, cg_mapped_xtc, 
                   aa_gro, aa_xtc,
                  size='900px'):
    """
    Render an overlay of an atomistic trajectory and its coarse-grained mapping.

    This function loads two trajectories into an NGL viewer:
    an atomistic (AA) system and a mapped coarse-grained (CG) system.
    The AA system is displayed as ball-and-stick, while the CG system
    is displayed as spheres representing beads. 

    Parameters
    ----------
    cg_mapped_gro : str or Path
        GRO file containing the topology/coordinates of the mapped CG system.
    cg_mapped_xtc : str or Path
        XTC trajectory corresponding to the mapped CG system.
    aa_gro : str or Path
        GRO file containing the topology/coordinates of the atomistic system.
    aa_xtc : str or Path
        XTC trajectory corresponding to the atomistic system.
    size : str, optional
        Size of the NGL viewer widget (CSS format), e.g. "900px".
        The viewer is rendered as a square of this size.

    Returns
    -------
    nglview.NGLWidget
        Interactive viewer containing the AA and CG trajectories overlayed.

    """
    
    cg_mapped_gro = Path(cg_mapped_gro).resolve()
    cg_mapped_xtc = Path(cg_mapped_xtc).resolve()
    aa_gro = Path(aa_gro).resolve()
    aa_xtc = Path(aa_xtc).resolve()
    
    view = nv.NGLWidget()
    u_aa = md.Universe(str(aa_gro), str(aa_xtc))
    u_cg = md.Universe(str(cg_mapped_gro), str(cg_mapped_xtc))

    view = nv.NGLWidget()
    view.add_trajectory(u_aa)
    view.add_trajectory(u_cg)

    # clear defaults
    view[0].clear_representations()
    view[1].clear_representations()
    
    # atomistic molecule
    view[0].add_representation("ball+stick", selection="NOT DOT")

    # CG molecule
    view[1].add_representation("spacefill", selection="NOT DOT", opacity=1.0, radius=.9)

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(frame=True, trim=True, transparent=True, factor=6)

    return view

def render_connely_surface(SASA_folder='./SASA', size='900px'):
    """
    Render overlapping Connolly (SASA) surfaces for atomistic and coarse-grained models.

    This function loads two structures containing precomputed SASA dot surfaces
    (typically generated with `gmx sasa`). The atomistic (AA) and coarse-grained (CG)
    systems are displayed together to visually compare their connely surfaces.

    Parameters
    ----------
    SASA_folder : str or Path, optional
        Directory containing the SASA surface files. The function expects
        the following structure:

            SASA_folder/
                AA/surface.pdb
                CG/surface.pdb

        Each `surface.pdb` must contain the molecular coordinates and
        pseudo-atoms with residue name `DOT` representing the sampled
        Connolly surface.
    size : str, optional
        Size of the NGL viewer widget (CSS format), e.g. "900px".
        The viewer is rendered as a square of this size.

    Returns
    -------
    nglview.NGLWidget
        Interactive viewer displaying the AA and CG structures with their
        corresponding SASA dot surfaces.

    Notes
    -----
    - AA atoms are shown as ball-and-stick.
    - CG beads are shown as spheres.
    - AA surface dots are colored blue.
    - CG surface dots are colored red.
    """

    SASA_folder = Path(SASA_folder).resolve()

    view = nv.NGLWidget()
    
    # component 0: atomistic structure + its DOT surface
    view.add_component(f"{SASA_folder}/AA/surface.pdb")
    # component 1: CG structure + its DOT surface
    view.add_component(f"{SASA_folder}/CG_Mapped/surface.pdb")
    
    # clear defaults
    view[0].clear_representations()
    view[1].clear_representations()
    
    # atomistic molecule
    view[0].add_representation("ball+stick", selection="NOT DOT")
    view[0].add_representation(
        "spacefill",
        selection="DOT",
        radius=0.11,
        opacity=0.35,
        color="blue"
    )
    # CG molecule
    view[1].add_representation("spacefill", selection="NOT DOT", opacity=1.0, radius=.9)
    view[1].add_representation(
        "spacefill",
        selection="DOT",
        radius=0.11,
        opacity=0.45,
        color="red"
    )
    view.center()

    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(frame=True,trim=True,transparent=True,factor=6)
    return view
                         
def render_mapping_vmd(cg_mapped_gro, cg_mapped_xtc, aa_gro, aa_xtc,
                       vmd='vmd', vmd_resolution=1000, dir_out='.', 
                       render=True, mogrify=True, plot=True):
    '''
    Visualize and render the mapping between atomistic and coarse-grained models.

    This function generates a VMD visualization showing the coarse-grained
    mapping overlaid on top of the corresponding atomistic structure. A
    template VMD state is modified to load the provided trajectories and
    optionally used to render images from multiple viewpoints.

    The workflow consists of:
    1. Preparing a VMD visualization state using the provided AA and CG files.
    2. Optionally running VMD in text mode to render images.
    3. Optionally converting rendered images to PNG using ImageMagick.
    4. Optionally assembling the rendered images into a comparison figure.

    Parameters
    ----------
    cg_mapped_gro : str or Path
        Structure file of the mapped coarse-grained system (.gro or .pdb).
    cg_mapped_xtc : str or Path
        Trajectory file of the mapped coarse-grained system (.xtc or .trr).
    aa_gro : str or Path
        Atomistic structure file (.gro or .pdb).
    aa_xtc : str or Path
        Atomistic trajectory file (.xtc or .trr).
    vmd : str, optional
        Path to the VMD executable. Default assumes `vmd` is available in PATH.
    vmd_resolution : int, optional
        Resolution used when rendering images with VMD. Default is 1000.
    dir_out : str or Path, optional
        Output directory where render files and images will be written.
    render : bool, optional
        If True, run VMD in text mode to generate render images.
    mogrify : bool, optional
        If True, convert rendered `.tga` images to `.png` using ImageMagick.
    plot : bool, optional
        If True, assemble rendered images into a matplotlib figure.

    Notes
    -----
    Requires:
    - VMD installed and accessible via `vmd`
    - ImageMagick installed for `mogrify` when `mogrify=True`
    
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
    dir_writing = dir_out / "Renders"
    dir_writing.mkdir(parents=True, exist_ok=True)
    
    ## Fix vis
    with open(vmdvis) as topinput:
        top = topinput.readlines()
        
    ##Careful! These are hardcoded. Will make this better at some point.
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
        subprocess.run(["mogrify", "-format", "png", "-transparent", "white", "*.tga"]
                        , check=True, cwd=dir_writing)    
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)
        views = ["MAPzz", "MAPyy", "MAPxx"]
        img0 = mpimg.imread(dir_writing / f"{views[0]}.png")
        x_min, x_max, y_min, y_max = _calculate_non_transparent_bounds(img0)
        for ax, v in zip(axs, views):
            img = mpimg.imread(dir_writing / f"{v}.png")
            ax.imshow(img)
            ax.set_xlim(x_min-100, x_max+100)
            ax.set_ylim(y_max+100, y_min-100)
            ax.set_axis_off()

        fig.savefig(f'{dir_writing}/Mapping.png', dpi=300, transparent=True, bbox_inches='tight')


def render_connely_surface_vmd(vmd='vmd', vmd_resolution=1000, dir_out='.', 
                               render=True, mogrify=True, plot=True):
    '''
    Render Connolly surface images using VMD and optionally assemble a figure.

    This function copies VMD visualization scripts distributed with SHAKER
    (`surface.vmd` and `surface_render.vmd`) into an output directory and
    optionally runs VMD in text mode to render images. Rendered `.tga` images
    can be converted to `.png` using ImageMagick, and multiple viewpoints can
    be combined into a single matplotlib figure.

    Parameters
    ----------
    vmd : str, optional
        Path to the VMD executable (default: "vmd", assuming it is on PATH).
    vmd_resolution : int, optional
        Rendering resolution passed to VMD via `-size` (default: 1000).
    dir_out : str or Path, optional
        Output directory. Render assets are written to `dir_out/Renders/`.
    render : bool, optional
        If True, run VMD in text mode to generate render images.
    mogrify : bool, optional
        If True, convert `.tga` renders to `.png` using ImageMagick `mogrify`.
    plot : bool, optional
        If True, load the rendered PNGs and create a composite figure.

    Notes
    -----
    Requires:
    - VMD installed and accessible via `vmd`
    - ImageMagick installed for `mogrify` when `mogrify=True`

    '''

    ## Fix pathing
    dir_out = Path(dir_out).resolve()
    
    ## Prepare render dir.
    dir_writing = dir_out / "Renders"
    dir_writing.mkdir(parents=True, exist_ok=True)

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
        subprocess.run(["mogrify", "-format", "png", "-transparent", "white", "*.tga"]
                        , check=True, cwd=dir_writing)   
    if plot:
        fig, axs = plt.subplots(1,3, figsize=(10,5),tight_layout=True)

        views = ["SASAzz", "SASAyy", "SASAxx"]
        img0 = mpimg.imread(dir_writing / f"{views[0]}.png")
        x_min, x_max, y_min, y_max = _calculate_non_transparent_bounds(img0)
        for ax, v in zip(axs, views):
            img = mpimg.imread(dir_writing / f"{v}.png")
            ax.imshow(img)
            ax.set_xlim(x_min-100, x_max+100)
            ax.set_ylim(y_max+100, y_min-100)
            ax.set_axis_off()

        # Create fake legend handles
        blue_patch = mpatches.Patch(color='tab:blue', label='AA surface')
        red_patch  = mpatches.Patch(color='tab:red',  label='CG Mapped surface')
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

def _best_grid(n: int):
    """Return (nrows, ncols) for n plots using a near-square layout."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def _predict_figsize(grids, cell_w=3.2, cell_h=1.5, section_gap_h=0.6,
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
        