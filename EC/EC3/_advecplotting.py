#includes plotly based themes for advanced plots. 


def plotline3D():
    import pandas as pd
    import numpy as np
    import chart_studio.plotly as py
    import cufflinks as cf
    import seaborn as sns
    import plotly.express as px
    %matplotlib inline

    # Make Plotly work in your Jupyter Notebook
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected=True)
    # Use Plotly locally
    cf.go_offline()

    fig = px.line_3d(flights, x='year', y='month', z='passengers', color='year')
    fig
    
    