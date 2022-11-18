from matplotlib.ticker import MultipleLocator, FormatStrFormatter,AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#_______________QUICK PLOT MODIFIER ROUTINES_______________
class plot_modifiers():
    def __init__(self):
        return
    
    #_____Annotations, Latexing etc.______
    def annotate(self, fig = None, labels = 'abc', style = 'lower', parenthesis = False, size = 12, weight = 'bold', loc = (-0.13, 1.04), append = "", label_by = 'all '):
        """
        Annotate with a, b, c
        
        Args:
            label_by: 'row', 'col', 'all'
        """
        import string
        
        if fig == None: 
            fig = plt.gcf()
            
        if labels == 'abc':
            abc = list(string.ascii_lowercase)
            #abc = list(string.ascii_uppercase)
            labels = abc[:len(fig.axes)]
        
    
    
        
        
        for index, (label, ax) in enumerate(zip(labels, fig.axes)):
            ax.text(loc[0], loc[1], label+append,
            transform=ax.transAxes,
            size=size, weight=weight)
        return
    
    def ltx_str(self, sym, which = None):
        """
        Convenience function for returning common latex strings.
        
        For greek letters sym should be abbreviation/acronym e.g. 'sr'
        
        Can also use names e.g. sym can be 'I' or 'current'
        
        
        Args:
            sym: symbol e.g. 'I', 'V'
            which: subtype, e.g. 'specific', 'areal'
        """
        
        if sym == 'I' or sym.lower() =='current':
            if which == 'specific': ret = 'I$_\mathrm{sp}$(Ag$^{-1})$'
            if which == 'areal': ret = 'I$_\mathrm{sp}$(Ag$^{-1})$##########'
            
        if sym.lower() == 'sr':
            if which == None: ret = r'$\nu$ (mVs$^{-1}$)'
            if which == 'sqrt': ret = r'$\nu^{1/2}$ (mVs$^{-1})^{1/2}$'
        

        return ret
    
    
    def create_tag(self, ax, pos = (0.017, 0.95), name = None, 
                   box_clr = 'red', edgecolor = 'black', alpha = 0.6, boxstyle = 'round', 
                   fontcolor = 'black', alignment = ('left', 'top')):
        """
        Purpose: 
            Create a tag at the top left corner of an axis. Use position to change x and y.
        """
        ax.text(pos[0], pos[1], s = name, 
                    bbox=dict(facecolor=box_clr, edgecolor = edgecolor, alpha=alpha, boxstyle=boxstyle), 
                    color = fontcolor, horizontalalignment=alignment[0], verticalalignment=alignment[1],transform=ax.transAxes, )
        
        return
    
    #______________Axes_________
    def ax_divider(self, ax, divisions = 4, size =(1.5, 1), direction = "bottom", pad = 0):
        """
        Divides vertically if no zip.
        Usage:
            axs = pm.ax_divider(ax0, divisions = 5)


        Args:
            direction: direction in which to add axes: "top", "bottom", "right", "left"
        Returns:
            Axes
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1 import Divider
        divider = make_axes_locatable(ax)

        axes = [ax]
        for i in range(divisions-1):
            ax2 = divider.append_axes(direction, size="100%", pad=pad)
            axes.append(ax2)

        return axes 
    
    #_____________Legend Ordering_______
    def legend_order(self, ax, order = None):
        """
        Reorder legend
        Untested
        """
        handles, labels = ax.get_legend_handles_labels()
        print(order)
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        
        return

    def scientific_labels(self, ax, pos = 0.075, axis = 'both'):
        """
        axis = 'x', 'y', 'both'
        """
        
        
        ax.ticklabel_format(axis=axis, style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=True)
        ax.get_yaxis().get_offset_text().set_x(pos)

        return
class plot_makers():
    """
    Convenience class to create plots for gridding.
    """
    def __init__(self):
        
        return
    def create_fig(self, figsize = (19,19), tight_layout = True, shape = (3,3), order = 'row'):
        """
        Usage:
            Use with subplot2grid.
            1. Creates a figure,
            2. Divides it into shape
            3. Creates an iterable of order.
            
        Args:
            order: Could be a list of tuples signifiying x and y locations of the plots in the sequence in which to draw them.
            
        Returns:
            fig: figure object
            loc: an iter of order.
            shape: shape of the grid
            
        Examples:
            fig, loc, shape = create_fig(figsize = (19,19), shape = (4,4), order = 'row')
            
            fig, loc, shape = create_fig(figsize = (19,19), shape = (4,4), order = [(0,0), (0,1), (2,3)])

        """

        fig = plt.figure(figsize = figsize
                         , tight_layout = tight_layout
                        )
        if isinstance(order, str):
            if order.lower() == 'row':
                order = [(i,j) for i in range(0,shape[0]) for j in range(0,shape[1])]
            elif order.lower() == 'column':
                order = [(j,i) for i in range(0,shape[0]) for j in range(0,shape[1])]
            elif order.lower() == 'diagonal':
                order = [(i,i) for i in range(0,shape[0])]

        loc = iter(order)

        return fig, loc, shape    
class plot_presets():
    def __init__(self):
        return
    
    def modLegendDic(modify):
        """
        Return a standard legend dictionary EXCEPT for the values shown. 
        
        """
        default_legend = dict(frameon = False, loc = 2, prop = {'size':10}, ncol = 1)
        
        
        # if modify == None: 
        #     return default_legend
        default_legend.update(modify)
    
    
        
        
        

        return default_legend
    def modMarkerDic():
        default_marker = dict(marker='.', 
                    linestyle='--', 
                    color = 'brown',
                    markersize=1,
                    markeredgecolor="brown", 
                    mew=1,
                    #color='brown', 
                    markerfacecolor='white', 
                    fillstyle = 'full', 
                    #markerfacecoloralt='white',
                            )      
        return
    
    
#___________Wrappers___________
def plotcv(ax, x, y, xlabel = "Voltage (V) vs. Hg/HgO", ylabel = r"I (mA)", 
           xlim = None, ylim = None, legend = None,  
           cycler = None,  marker_dic = dict(), legend_dic = dict(), 
           **kwargs):
    
    plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs) 
    
    return

def ploteis(ax, x, y, xlabel=r"Z' ($\Omega$)",ylabel =r"-Z'' ($\Omega$)", z_im_neg = True, xlim = None, ylim = None,legend = None, cycler = None, marker_dic = dict(), legend_dic = dict(), **kwargs):
    if z_im_neg:
         plotec(ax, x, -y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs) 
    else:
         plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs) 
    
    
    return
     
def plotgcd(ax, x, y, xlabel = "Time (s)", ylabel ="Voltage (V) vs. Hg/HgO", xlim = None, ylim = None, legend = None,  cycler = None,  marker_dic = dict(), legend_dic = dict(), **kwargs):
    plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs)
    return

def plotbvalue(ax, x, y, xlabel ="$log (sr)$" , ylabel ="$log (i_p)$", xlim = None, ylim = None, legend = None, cycler = None,  marker_dic = dict(), legend_dic = dict(), **kwargs):
    plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs)
    return

def plotpeaks(ax, x, y, xlabel = "$sr^{1/2}$", ylabel ="$i_{pa}$", xlim = None, ylim = None, legend = None, cycler = None, marker_dic = dict(), legend_dic = dict(), **kwargs):
    plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs)
    return

def plotRagone(x, y, xlabel = "Energy", ylabel = "Power", xlim = None, ylim = None, legend = None, cycler = None, marker_dic = dict(), legend_dic = dict(), **kwargs):
    #plotec(ax, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, plot = 'scatter', **kwargs)
    #Use seaborn dataframe for hue, size, style. 
    #Use plotly
    
    """
    Information on plot:
    x = Energy
    y = Power
    Color = Group
    Shape = Material
    Size = Capacity
    
    For Plotly:
    Dropdown menu = Scan Rate
    
    
    
    """

    
    sns.scatterplot(data = df, x = 'Energy', y = 'Power', hue = 'Group', style = 'Material', s = 'Capacity',
                   sizes = (20,200))
    
    # mapping_dic = dict([(e[0]+" - "+e[1], e[0]) for e in list((st.mats.index))])
    # QR().group_hue_and_style_from_dict(sp.axes, second_category = 'Material', mapping_dic = mapping_dic)
    
    
    return
#__________________________________________

def abline(slope, intercept):
    """
    Plot a line from slope and intercept
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
def plotinset(ax, x, y, locator = [0.65, 0.65, 0.3, 0.3], xlabel = None, ylabel =None, xlim = None, ylim = None, legend = None, cycler = None, marker_dic = dict(), legend_dic = dict(), **kwargs): 
    #EIS_Inset
    axins = ax.inset_axes(locator) 
    
    plotec(axins, x, y, xlabel, ylabel, xlim, ylim, legend = legend, cycler = cycler,  marker_dic = marker_dic, legend_dic = legend_dic, **kwargs)
    axins.locator_params(axis='y', nbins=2)
    axins.locator_params(axis='x', nbins=3)
    
    #axins.yaxis.set_major_formatter('{x:0<5.1f}')
    
    axins.xaxis.set_minor_locator(AutoMinorLocator(1))
    axins.yaxis.set_minor_locator(AutoMinorLocator(1))
    axins.tick_params(axis='both', which='major', pad=2.5)
    axins.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useOffset=None, useLocale=None, useMathText=None)
    #axins.margins(x=0, y=0)
    axins.set_xlabel(axins.get_xlabel(), fontdict=None, labelpad=0, loc=None)
    axins.set_ymargin(0.51)   
    axins.set_xmargin(0.1)   


    
def ploteis_multiple(ax):
    """
    Better to merge with EIS fits or break down into wrapper functions or give the option to plot fits as well. 
    
    impedance.py returns Complexes. Simply split them up and use that. Keep this as an alternate. 
    Or simply as a way to render the bode plot. 
    
    
    """
    # %% Plot all circuits - Nyquist

    from impedance.visualization import plot_nyquist

    fig, ax = plt.subplots(figsize=(5,5))
    plot_nyquist(ax, Z)
    plot_nyquist(ax, randles_fit, fmt='--')
    plot_nyquist(ax, randlesCPE_fit, fmt='-.')
    #plot_nyquist(ax, customCircuit_fit, fmt='-')
    #plot_nyquist(ax, customConstantCircuit_fit, fmt='-')
    ax.legend(['Data', 'Randles', 'Randles w/ CPE', 'Custom Circuit'])
    plt.show()

    # %% Plot all circuits - Bode

    randles.plot(f_data=frequencies, Z_data=Z, kind='bode')
    randlesCPE.plot(f_data=frequencies, Z_data=Z, kind='bode')
    #circuit.plot(f_data=frequencies, Z_data=Z, kind='bode')
    plt.show()

    # %% Print circuit info
    print(randlesCPE)

    # %% Save data
#     filename = "EIS-FittedVals-Model1.csv"
#     impedances = randlesCPE_fit
#     preprocessing.saveCSV(filename, frequencies, impedances)

#     filename = "EIS-FittedVals-Model2.csv"
#     impedances = randles_fit
#     preprocessing.saveCSV(filename, frequencies, impedances)


    # %% Plotting
    import matplotlib.pyplot as plt
    from impedance.visualization import plot_nyquist

    fig, ax = plt.subplots()
    plot_nyquist(ax, Z, fmt='o')
    plot_nyquist(ax, Z_fit, fmt='-')



    plt.legend(['Data', 'Fit'])
    plt.show()

    circuit.plot(f_data=frequencies, Z_data=Z, kind='bode')
    plt.show()

    
def plotreg_param(ax:tuple, x, y, xlabel = "Voltage (V) vs. Hg/HgO", ylabel ="Regression Parameter", xlim = None, ylim = None, legend = 'default',  cycler = None,  marker_dic = dict(), legend_dic = dict(), annotate = True, **kwargs):
    """
    Usage: 
        Define two axes in the main code and pass them as a tuple:
        ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2)    
        
        x: Series (NOT Dataframe) of voltages.
        y: Dataframe of regression parameters. 


    Code:
        Define vol (ax)
        Break it into two 
            Define k1k2_ch and k1k2_disch
        Plot

    Creates: 
    - Two axes with a shared y-axis.
    
    """
    #____Settings____
    ax[1].sharey(ax[0])

    ax[0].patch.set_facecolor('#EEEED099')
    ax[1].patch.set_facecolor('#AAEEEE99')
    #ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    ax[1].get_yaxis().set_visible(False)

    

    plt.suptitle(xlabel, y = 0.45, fontsize=12)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    #______________________
    
    xlim = [0, round(x.to_frame().max()[0],2)]
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter,AutoMinorLocator
    
    if y.shape[1] == 3: 
        #___Reorder data so that it's B0, B1, B2___
        y = y.copy()
        c = y.columns
        y = y.reindex(columns=[c[1],c[2],c[0]])
        if legend == 'default':
            legend = [r"$\mathrm{\beta_{1}}$", r"$\mathrm{\beta_{2}}$",r"$\mathrm{\beta_{3}}$"]
    if y.shape[1] == 2: 
        if legend == 'default': legend = ['k1', 'k2']
            
    
    switch = int(x.shape[0]/2)
    x_ch, y_ch, x_dis, y_dis = x[0:switch], y[0:switch], x[switch:], y[switch:]
    
    
    plotec(ax[1], x_dis, y_dis, xlabel = xlabel, ylabel = ylabel, xlim = xlim[::-1], ylim = ylim, cycler = cycler, marker_dic = marker_dic, **kwargs)
    plotec(ax[0], x_ch, y_ch, xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim, legend = legend, cycler = cycler, marker_dic = marker_dic, legend_dic = legend_dic, **kwargs)
    


    #____Annotate___
    if annotate:
        ax[0].text(0.01, 0.98, ' Forward Scan', transform=ax[0].transAxes
                        ,fontsize=14
                        ,fontweight='bold'
                        , va='top'
                        , ha = 'left')
        ax[1].text(0.99, 0.98, 'Reverse Scan', transform=ax[1].transAxes
                        ,fontsize=14
                        ,fontweight='bold'
                        , va='top'
                        , ha = 'right')    

        
def plotcvresiduals2(ax, df, x = None, y = None, hue = None, xlim = None, ylim = None, legend = 'default',  cycler = None,  marker_dic = dict(), legend_dic = dict(), annotate = True, **kwargs):
    """
    Uses matplotlib instead of facetgrid to plot cv residuals. 
    """
    
    return
    
def plotcvresiduals(df, **kwargs):

    #___Preset for plotting in case there are no kwargs___
    fontsize = 12
    if len(kwargs) == 0:
        kwargs = {'height': 2, 'aspect': 2.5, 'sharey': True, 'margin_titles':True, 'legend_out':False}
    sns.set_theme(style="white")    
    plt.rcParams['xtick.major.size'] = 20
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    #__________________________________
    
    
    sr_lab = "SR"

    #formatted_df = pd.melt(df,["Voltage", "Scan"],var_name=sr_lab, value_name="Residual")
    
    formatted_df = df.copy()
    formatted_df['Residual'] *= 10**3
    vol_fac = formatted_df['Voltage'].unique()
    
    
    g = sns.FacetGrid(formatted_df, row="Voltage", **kwargs)
    g.map_dataframe(sns.scatterplot, x="SR", y="Residual", hue = "Scan", alpha = 0.8, s = 75,palette=['#F14040', '#37AD6B'])
    g.fig.tight_layout()
    
    sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    
    plt.subplots_adjust(hspace = 0)
    g.add_legend(ncol = 2, frameon = False)
    
    axes = g.axes.flatten()
    axes[0].set_ylabel("") 
    axes[1].set_ylabel("Residual (mA)") 
    axes[2].set_ylabel("")
    
    sr_lab = plot_modifiers().ltx_str('sr')
    axes[2].set_xlabel(sr_lab)

    for ax in axes:
        ax.tick_params(axis='both', which = 'major', direction = 'in', length = 6, width = 1, labelsize = fontsize)
        ax.tick_params(axis='both', which = 'minor', direction = 'in', length = 4, width = 1, labelsize = fontsize)
        ax.locator_params(axis='y', nbins=3)



    return


def images_to_grid(folder:str = None, grid = (2,2), figsize = (12,8), labels = 'auto', tight_layout = True):
    """
    Works best for figures of equal dimensions.
    
    Takes folder path. 
    """
    rows, columns = grid

    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    if labels == 'auto': 
        labels = list(string.ascii_lowercase)[0:len(onlyfiles)]
        labels = [l+')' for l in labels]


    fig = plt.figure(figsize=figsize, tight_layout = tight_layout)
    for i, f in enumerate(onlyfiles):
        #print(i)
        fig.add_subplot(rows, columns, i+1)


        image = plt.imread(f)
        plt.imshow(image)
        plt.axis('off')
        plt.title(labels[i])
        
    
def folder_images_to_grid(folder = None, figsize = (None, None), grid = (2,2)):
    """
    Untested & Incomplete

    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np


    from os import listdir
    from os.path import isfile, join

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    for f in onlyfiles:
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)

        # showing image
        plt.imshow(f)
        plt.axis('off')
        plt.title("First")



    #__________________________

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)


    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
        #____________________________

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

    
def set_ax_size(w,h, ax=None):
    """ 
    w, h: width, height in inches 
    
    Usage:
        fig, ax=plt.subplots()
        ax.plot([1,3,2])
        set_size(5,5)
        plt.show()
    
    """
    
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def plotec(ax, x, y, xlabel = None, ylabel =None, xlim = None, ylim = None, legend = None, cycler = None, marker_dic = dict(), legend_dic = dict(), plot = 'plot', **kwargs):
    """
    Mostly self-explanatory. 
    
    Remember: Marker overrides cycler. 
    
    Args:
    plot: 'plot', 'scatter'
    **kwargs may include:
        - set_aspect
        - set_size_inches
        - fontsize: Uses same fontsize for both labels as well as titles
    
    """
    #____Presets____
    fontsize = 12 #For labeling.
    
    #____Handle possible kwargs___
    if 'set_aspect' in kwargs:
        print(kwargs['set_aspect'])
        #ax.set_aspect(kwargs['set_aspect'])
        plt.gca().set_aspect(kwargs['set_aspect'])

    
    if 'set_size_inches' in kwargs:
        plt.gcf().set_size_inches(kwargs['size']) 
    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']
        
        
    
    
    #_____________________________
            
    

    
    if cycler:
        ax.set_prop_cycle(cycler)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    
    ax.locator_params(axis='both', tight=True, nbins = 4)
    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='x', nbins=4)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    
    ax.tick_params(axis='both', which = 'major', direction = 'in', length = 6, width = 1, labelsize = fontsize)
    ax.tick_params(axis='both', which = 'minor', direction = 'in', length = 4, width = 1, labelsize = fontsize)
    

    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])    
    
    x = np.array(x)
    y = np.array(y)
    if plot =='plot': ax.plot(x, y,**marker_dic)
    if plot =='scatter': ax.scatter(x, y,**marker_dic)
    
    if legend:
        ax.legend(legend, labelspacing = 0, **legend_dic 
                 )
    

        


    
    return

def percent_capacitive(ax, srates, y1, y2 = None, barWidth = 1, palette = ('r', 'g')):
    """
    Percent capacitive plot
    Incomplete.
    """
    
    br1 = srates
    

    plt.bar(br1, y1
    , color =palette[0], width = barWidth,
            edgecolor ='grey', label ='IT')
    if y2:
        br2 = [x + barWidth for x in br1]
        plt.bar(br2, y2
        , color =palette[1], width = barWidth,
                edgecolor ='grey', label ='ECE')
    return

#__________________Study Plots_________________


def plot_study_peaks_vs_sqrtv():
    
    pass

class studyplots():
    def __init__(self):
        return
    
        
    def plotly_cvs(self, st):
        import numpy as np
        import plotly.graph_objs as go
        from itertools import cycle

        def plotly_color_map(names):
            # From https://stackoverflow.com/a/44727682
            plotly_colors = cycle(['#1f77b4',  # muted blue
                                   '#ff7f0e',  # safety orange
                                   '#2ca02c',  # cooked asparagus green
                                   '#d62728',  # brick red
                                   '#9467bd',  # muted purple
                                   '#8c564b',  # chestnut brown
                                   '#e377c2',  # raspberry yogurt pink
                                   '#7f7f7f',  # middle gray
                                   '#bcbd22',  # curry yellow-green
                                   '#17becf'  # blue-teal
                                   ])

            return dict(zip(names, plotly_colors))

        import plotly.io as pio
        pio.renderers.default = "browser"

        features = []
        labels = []
        for i, m in enumerate(st.mats['Object']):
            y = m.cv.currents.iloc[:,0]
            x = m.cv.voltages.iloc[:,0]
            d = np.array(pd.concat([x,y], axis =1)).tolist()
            features.append(d)

            #labels.append(m.name*x.shape[0])

        legend_groups = [m.name for m in st.mats['Object']]

        traces = [False if (len(legend_groups[:i])>0 and l in legend_groups[:i]) 
                  else True for i, l in enumerate(legend_groups)]

        cm = plotly_color_map(set(legend_groups))

        fig = go.Figure()
        for i, feat in enumerate(features):
            feat = np.array(feat)
            fig.add_trace(
                go.Scatter3d(

                    x=([i]*len(feat)),
                    y=feat[:,0], #Voltages
                    z=feat[:,1], #currents


                    mode='lines',
                    line={"color":cm[legend_groups[i]]},
                    legendgroup=legend_groups[i],
                    hovertext=legend_groups[i],
                    showlegend=traces[i],
                    name="label_{}".format(legend_groups[i])
                )

            )
            fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="LightSteelBlue",)

        fig.show()

        return

    def plot_study_reg_params(self, df):
        """
        Plot regression parameters by study group. 
        #Untested
        """

        shape = (len(list(df.index.get_level_values(0).unique())), 2)


        fig = plt.figure()
        clrs_list=['k','b','g','r', 'c'] # list of basic colors
        styl_list=['-','--','-.',':', 'loosely dotted'] # list of basic linestyles

        for j, (ind, group) in enumerate(df.groupby(level=0)):
            k = 2*j # For gap, use k.

            clrr = clrs_list[j % 5]
            ax0 = plt.subplot2grid(shape, (j,0), rowspan=1, colspan=1);
            ax1 = plt.subplot2grid(shape, (j,1), rowspan=1, colspan=1);

            legend = group.index
            for i, m in enumerate(group['Object']):
                styl=styl_list[i % 4]

                ecp.plotreg_param((ax0, ax1), m.cv.voltages.iloc[:,0].to_frame(), m.cv.quad_reg_params.iloc[:,0].to_frame(),
                                 marker_dic = dict(color = clrr, ls = styl),
                                  legend_dic = {'loc': 3, 'ncol':3},
                                  legend = legend

                                 )



        return
    
    



        