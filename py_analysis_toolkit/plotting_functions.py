import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import os
from os import path
from matplotlib.ticker import (MultipleLocator, FixedLocator, FixedFormatter, FormatStrFormatter, FuncFormatter,
							   AutoMinorLocator,LogLocator)

colors = np.array([[0,115,179],
				   [230,160,37],
				   [204,121,167],
				   [79,111,52],
				   [223,197,153],
				   [153,179,223],
				  ])/255

def set_plotting_options():
	""" Set matplotlib plotting settings """
	colmap = 'seismic'
	facecol = '#d88f8f'

	plt.rcParams['text.usetex'] = True
	# `plt.rcParams['text.latex.preamble'] = [r'\newcommand{\sym}[1]{\textit{#1}}',
	# 									   r'\newcommand{\dIdV}{d\textit{I}/d\textit{V} (2e\textsuperscript{2}/h)}',
	# 									   r'\newcommand{\sub}[1]{\textsubscript{#1}}']`

	# mpl.rcParams["font.family"] = 'Arial'
	mpl.rcParams["font.family"] = "sans-serif"
	mpl.rcParams["font.sans-serif"] = ['DejaVu Sans']
	mpl.rcParams["axes.titlesize"] = 7
	mpl.rcParams["axes.labelsize"] = 10
	mpl.rcParams["xtick.labelsize"] = 10
	mpl.rcParams["ytick.labelsize"] = 10
	mpl.rcParams['legend.fontsize'] = 7
	mpl.rcParams["font.size"] = 7
	mpl.rcParams["figure.figsize"] = [5.04,6.3]
	mpl.rcParams["figure.dpi"] = 180
	mpl.rcParams["ytick.major.size"] = 2.835
	mpl.rcParams["ytick.major.width"] = 1
	mpl.rcParams["ytick.minor.size"] = 2.835
	mpl.rcParams["ytick.minor.width"] = 1
	mpl.rcParams["ytick.direction"] = "out"
	mpl.rcParams["xtick.major.size"] = 2.835
	mpl.rcParams["xtick.major.width"] = 1
	mpl.rcParams["xtick.direction"] = "out"
	mpl.rcParams["xtick.minor.size"] = 2.835
	mpl.rcParams["xtick.minor.width"] = 1
	mpl.rcParams["savefig.bbox"] = None
	mpl.rcParams["savefig.pad_inches"] = 0.0
	mpl.rcParams["figure.subplot.left"] = 0.15
	mpl.rcParams["figure.subplot.right"] = 0.9
	mpl.rcParams["figure.subplot.top"] = 0.9 #0.98
	mpl.rcParams["figure.subplot.bottom"] = 0.15
	mpl.rcParams["axes.linewidth"] = 0.8
	mpl.rcParams["legend.fontsize"] = 7
	mpl.rcParams["legend.loc"] = "upper right"
	mpl.rcParams["lines.markersize"] = 3
	mpl.rcParams["lines.linewidth"] = 1
	mpl.rcParams['pdf.fonttype'] = 42 #Exports text as font not as vector
	# mpl.rcParams['mathtext.rm'] = 'Arial'
	# mpl.rcParams['mathtext.default'] = 'it'


def set_plotting_options_lockin_mapping():
	""" Set matplotlib plotting settings for lockin_mapping notebook """
	colmap = 'seismic'
	facecol = '#d88f8f'

	plt.rcParams['text.usetex'] = False
	plt.rcParams['text.latex.preamble'] = [r'\newcommand{\sym}[1]{\textit{#1}}',
										  r'\newcommand{\dIdV}{d\textit{I}/d\textit{V} (2e\textsuperscript{2}/h)}',
										   r'\newcommand{\sub}[1]{\textsubscript{#1}}'
										  ]

	mpl.rcParams["font.family"] = "sans-serif"
	mpl.rcParams["font.sans-serif"] = ['DejaVu Sans']
	mpl.rcParams["axes.titlesize"] = 7
	mpl.rcParams["axes.labelsize"] = 7
	mpl.rcParams["xtick.labelsize"] = 7
	mpl.rcParams["ytick.labelsize"] = 7
	mpl.rcParams['legend.fontsize'] = 7
	mpl.rcParams["font.size"] = 7
	mpl.rcParams["figure.figsize"] = [5.04,6.3]
	mpl.rcParams["figure.dpi"] = 180
	mpl.rcParams["ytick.major.size"] = 2.835
	mpl.rcParams["ytick.major.width"] = 0.5
	mpl.rcParams["ytick.minor.size"] = 2.835*0.5
	mpl.rcParams["ytick.minor.width"] = 0.5
	mpl.rcParams["ytick.direction"] = "out"
	mpl.rcParams["xtick.major.size"] = 2.835
	mpl.rcParams["xtick.major.width"] = 0.5
	mpl.rcParams["xtick.direction"] = "out"
	mpl.rcParams["xtick.minor.size"] = 2.835*0.5
	mpl.rcParams["xtick.minor.width"] = 0.5
	mpl.rcParams["savefig.bbox"] = None
	mpl.rcParams["savefig.pad_inches"] = 0.0
	mpl.rcParams["figure.subplot.left"] = 0.15
	mpl.rcParams["figure.subplot.right"] = 0.9
	mpl.rcParams["figure.subplot.top"] = 0.9
	mpl.rcParams["figure.subplot.bottom"] = 0.15
	mpl.rcParams["axes.linewidth"] = 0.5
	mpl.rcParams["legend.fontsize"] = 7
	mpl.rcParams["legend.loc"] = "upper right"
	mpl.rcParams["legend.frameon"] = False
	mpl.rcParams["legend.handlelength"] = 1
	mpl.rcParams["legend.borderaxespad"] = 0.2
	mpl.rcParams["lines.markersize"] = 3
	mpl.rcParams["lines.linewidth"] = 1
	mpl.rcParams['pdf.fonttype'] = 42 #Exports text as font not as vector

	colors = np.array([[0,115,179],
					   [230,160,37],
					   [204,121,167],
					   [79,111,52],
					   [223,197,153],
					   [153,179,223],
					  ])/255
	colors = [tuple(c) for c in colors]
	plt.rc('axes', prop_cycle=(cycler('color', colors)))

def add_colorbar(fig, ax, mesh, ticks=None, width=0.15, pad_h=1, pad_w=0, height=1.5, loc='topright', label=r'\dIdV', 
				 labelsize=None,fontsize=15,outline_visible=False, labelpad=False):
	""" Function to add colorbar above right top of the axes.

	Parameters:
		ticks:  None: By default add ticks at the minimum and maximum of the colorscale.
				Supply array for custom ticks.
		width:  Width of the colorbar in units of the relative size of the figure
		pad_h:  Vertical distance between axis and colorbar in mm
		pad_w:  Horizontal distance between axis and colorbar in mm. 
				Defaults to 0 for loc='topright', or pad_h for loc='inside'.
		height: Height of the colorbar in mm
		loc:    Colorbar location (topright or inside)
		label:  Label to add left of the colorbar. Use False to turn off
		labelsize: Fontsize of the tick labels.

	Returns:
		cax:    Colorbar axis instance.
	"""
	
	def mm_to_rel(val):
		#Convert mm to relative size in the figure
		return val/(fig.get_size_inches()*25.4)
	
	pad_h = mm_to_rel(pad_h)[-1] # 为什么要有 [-1]???
	pad_w = mm_to_rel(pad_w)[-1]
	height = mm_to_rel(height)[-1]
	
	p = ax.get_position()
	if loc=='topright':
		cax = fig.add_axes((p.x1-width-pad_w, p.y1+pad_h, width, height)) #x,y,w,h
		cb=fig.colorbar(mesh, cax=cax, orientation='horizontal')
		cb.outline.set_visible(outline_visible)
		cax.xaxis.tick_top()
	
	elif loc=='inside':
		if pad_w == 0:
			pad_w = pad_h
		cax = fig.add_axes((p.x1-pad_w-width, p.y1-pad_h-height, width, height))
		fig.colorbar(mesh, cax=cax, orientation='horizontal')
	elif loc=='right':
		if width == 'Full_scale':
			size = p.y1-p.y0
			cax = fig.add_axes((p.x1+pad_w, p.y0, height, size))
			cb=fig.colorbar(mesh, cax=cax, orientation='vertical') 
			cb.outline.set_visible(outline_visible)              
		else:
			cax = fig.add_axes((p.x1+pad_w, p.y1-pad_h-width, height, width))
			cb=fig.colorbar(mesh, cax=cax, orientation='vertical')  
			cb.outline.set_visible(outline_visible)        
	        
	# Ticks defined as values on the colormap
	if loc=='right':
		if ticks==None:
			cax.yaxis.set_ticks([*mesh.get_clim()])
		else:
			cax.yaxis.set_ticks(ticks)
		def norm_tick_str(y,pos): #pos argument needed for some unclear reason
			return f'{y:.3g}'
		cax.yaxis.set_major_formatter(FuncFormatter(norm_tick_str))
		cax.tick_params('y',labelsize=labelsize)
	else:    
		if ticks==None:
			cax.xaxis.set_ticks([*mesh.get_clim()])
		else:
			cax.xaxis.set_ticks(ticks)
		def norm_tick_str(x,pos): #pos argument needed for some unclear reason
			return f'{x:.3g}'
		cax.xaxis.set_major_formatter(FuncFormatter(norm_tick_str))  # ??????? 为什么这里没有括号呢????????????
		if loc=='topright':
			cax.xaxis.set_tick_params(pad=0)
		cax.tick_params('x',labelsize=labelsize,width=1,length=3)

	if label:
		if loc=='right':
			cax.set_ylabel(label,labelpad=labelpad, size=fontsize)          
		else:
			cax.text(p.x1-1.05*(width)-pad_w-labelpad,p.y1+1.2*pad_h,label,ha='right',transform=fig.transFigure,fontsize=fontsize)

	return cax

def add_seg_colorbar(fig, ax,mesh,ticks=None,width=0.15,pad_h=1, pad_w=0, height=1.5, loc='topright', label=r'\dIdV', labelsize=None,fontsize=15,outline_visible=False, stringformatx='%.1f', extend=None, extendfrac=None, labelpad=None):
	""" Function to add colorbar above right top of the axes.

	Parameters:
		ticks:  None: By default add ticks at the minimum and maximum of the colorscale.
				Supply array for custom ticks.
		width:  Width of the colorbar in units of the relative size of the figure
		pad_h:  Vertical distance between axis and colorbar in mm
		pad_w:  Horizontal distance between axis and colorbar in mm. 
				Defaults to 0 for loc='topright', or pad_h for loc='inside'.
		height: Height of the colorbar in mm
		loc:    Colorbar location (topright or inside)
		label:  Label to add left of the colorbar. Use False to turn off
		labelsize: Fontsize of the tick labels.

	Returns:
		cax:    Colorbar axis instance.
	"""
	
	def mm_to_rel(val):
		#Convert mm to relative size in the figure
		return val/(fig.get_size_inches()*25.4)
	
	pad_h = mm_to_rel(pad_h)[-1] # 为什么要有 [-1]???
	pad_w = mm_to_rel(pad_w)[-1]
	height = mm_to_rel(height)[-1]
	
	p = ax.get_position()
	if loc=='topright':
		cax = fig.add_axes((p.x1-width-pad_w, p.y1+pad_h, width, height)) #x,y,w,h
		cb=fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=ticks, spacing='uniform', format=stringformatx, extend=extend,extendfrac=extendfrac) # only this line matters
		cax.tick_params('x',labelsize=labelsize,width=1,length=3)
		# cax.xaxis.set_major_formatter(FormatStrFormatter(stringformatx)) # 必须是normalize?? 很烦
		cb.outline.set_visible(outline_visible)
		cax.xaxis.tick_top()
	
	elif loc=='inside':
		if pad_w == 0:
			pad_w = pad_h
		cax = fig.add_axes((p.x1-pad_w-width, p.y1-pad_h-height, width, height))
		fig.colorbar(mesh, cax=cax, orientation='horizontal')
	elif loc=='right':
		if width == 'Full_scale':
			size = p.y1-p.y0
			cax = fig.add_axes((p.x1+pad_w, p.y0, height, size))
			cb=fig.colorbar(mesh, cax=cax, orientation='vertical') 
			cb.outline.set_visible(outline_visible)              
		else:
			cax = fig.add_axes((p.x1+pad_w, p.y1-width, height, width))
			cb=fig.colorbar(mesh, cax=cax, orientation='vertical')  
			cb.outline.set_visible(outline_visible)        
	        

	if label:
		if loc=='right':
			cax.set_ylabel(label,labelpad=-1)          
		else:
			cax.text(p.x1-1.05*(width)-pad_w-labelpad,p.y1+1.2*pad_h,label,ha='right',transform=fig.transFigure,fontsize=fontsize)

	return cax

def add_vcut_markers(ax, xs, s=4**2, h=0.05, color=None, ec='k', lw=0.25):
	""" Add markers on axes ax at position xs

	Parameters:
		s:     Marker size. Note that scatter uses the square of rcParams lines.markersize.
		h:     Y position of the marker in relative axis coordinates
		color: Marker color. Defaults to None, in which case the default color cycler is used.
		ec:    Edgecolor of the marker. Use 'face' to use the marker color.
		lw:    Linewidth of marker edge.    
	"""
	ylim = ax.get_ylim()
	y = ylim[0] + h*(ylim[1]-ylim[0])
	for i,x in enumerate(xs):
		if  color==None:
			ax.scatter(x,y,marker='^',s=s,edgecolor=ec,linewidth=lw)
		else:
			ax.scatter(x,y,marker='^',color=color[i],s=s,edgecolor=ec,linewidth=lw)
			
def savefig(fig,filename,directory, fmts=['.png'], dpi=300, border=False):
	""" Rasterize colormaps and save a figure

	Parameters:
		fig:    Figure instance to save.
		fid:    Filename to use to save the figure (without extension)
		fmts:   List of file formats to save the figure as.
		dpi:    DPI used to save the figure.
		border: (Boolean) If True, plot a red boundary around the figure (this is not present in the exported figure).
				Useful to manually tune the padding of the figure boundaries.
	"""
	# Formatting
	for ax in fig.get_axes():
		ax.set_rasterization_zorder(1)
	
	# Saving
	if os.path.isdir(directory) == False:
		os.mkdir(directory)

	for fmt in fmts:
		fig.savefig(directory+filename+fmt,dpi=dpi)
	
	# Draw boundary around figure to show figsize
	if border:
		bounds = plt.Rectangle((0,0),1,1,ec='r',fill=False,transform=fig.transFigure)
		fig.add_artist(bounds)

class PowerNormalize(mpl.colors.Normalize):
	""" Creates a colorbar with custom boundaries and middle. Use the result in the normalization option of plotter (norm).
	Inputs:
		midpoint: Define the middle of the colorbar in the units you're plotting in (e.g. G0 in 2e^2/h)
		vmin: Define lower boundary of the colorbar (optional)
		vmax: Define upper boundary of the colorbar (optional)
	Example:
		plt.pcolormesh(d.x1,d.y,d.G1,cmap='seismic', norm=data.MidpointNormalize(vmin=0,vmax=None,midpoint=0.4) )
	"""

	def __init__(self, vmin=None, vmax=None, gamma=1, midpoint=None, clip=False):
		if midpoint == None:
			midpoint = np.mean([vmin,vmax])
		self.gamma = gamma
		self.midpoint = midpoint
		mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# First scale to 0-1 range, then apply power gamma
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		res = np.ma.masked_array(np.interp(value, x, y))
		return res**self.gamma

def plot_errorbar(ax,x,y,y_err,alpha1=0.55,alpha2=0.25,facecolor='r'):
	"""
	Plot error as shaded region, indicating both the 1-sigma and the 2-sigma error.
	Inputs:
	x:      x-values over which to plot
	y:      y-values over which to plot
	y_err:  1-sigma error in y-value
	alpha1: Opacity 1-sigma bar
	alpha2: Opacity 2-sigma bar
	"""
    
	s1 = ax.fill_between(x, y-y_err, y+y_err, alpha=alpha1, facecolor=facecolor)
	s2 = ax.fill_between(x, y-2*y_err, y+2*y_err, alpha=alpha2, facecolor=facecolor)
    
def format_axes(ax,stringformatx='%.3g',stringformaty='%.3g',xmajor=0.1,xminor=False,ymajor=0.1,yminor=False):
	"""
	Format axes labels and tick positions
	Inputs (type):
	ax (axis object):        axis instance to format
	stringformatx (string):  format the labels on the x-axis, default: %.3g
	stringformaty (string):  format the labels on the y-axis, default: %.3g
	xmajor (float or False): if not False, x-axis major ticks are placed separated by this number, default: 0.1
	xminor (float or False): if not False, x-axis minor ticks are placed separated by this number, default: False
	ymajor (float or False): if not False, y-axis major ticks are placed separated by this number, default: 0.1
	yminor (float or False): if not False, y-axis minor ticks are placed separated by this number, default: False
	"""
	if xmajor:
		ax.xaxis.set_major_locator(MultipleLocator(xmajor))
	if xminor:
		ax.xaxis.set_minor_locator(MultipleLocator(xminor)) 
	if ymajor:
		ax.yaxis.set_major_locator(MultipleLocator(ymajor))
	if yminor:
		ax.yaxis.set_minor_locator(MultipleLocator(yminor))
	ax.xaxis.set_major_formatter(FormatStrFormatter(stringformatx))
	ax.yaxis.set_major_formatter(FormatStrFormatter(stringformaty))

	
def format_axes2(ax,stringformatx=False,stringformaty=False,xmajor=False,xminor=False,ymajor=False,yminor=False):
	"""
	Format axes labels and tick positions
	Inputs (type):
	ax (axis object):        axis instance to format
	stringformatx (string):  format the labels on the x-axis, default: %.3g
	stringformaty (string):  format the labels on the y-axis, default: %.3g
	xmajor (float or False): if not False, x-axis major ticks are placed separated by this number, default: 0.1
	xminor (float or False): if not False, x-axis minor ticks are placed separated by this number, default: False
	ymajor (float or False): if not False, y-axis major ticks are placed separated by this number, default: 0.1
	yminor (float or False): if not False, y-axis minor ticks are placed separated by this number, default: False
	"""
	if xmajor:
		ax.xaxis.set_major_locator(MultipleLocator(xmajor))
	if xminor:
		ax.xaxis.set_minor_locator(MultipleLocator(xminor)) 
	if ymajor:
		ax.yaxis.set_major_locator(MultipleLocator(ymajor))
	if yminor:
		ax.yaxis.set_minor_locator(MultipleLocator(yminor))
	if stringformaty:		
		ax.yaxis.set_major_formatter(FormatStrFormatter(stringformaty))
	if stringformatx:
		ax.xaxis.set_major_formatter(FormatStrFormatter(stringformatx))