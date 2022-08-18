import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import datetime as dt
import json
import geopandas as gpd
import numpy as np
import pandas as pd
import re
import os

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# module to handle the data collection & clean up
from get_data import get_data

print(os.getcwd())
print(os.listdir() )
# =============================================================
# Get the data
# =============================================================
send_requests = False
if send_requests:
    # retieve the data from the gov.uk coronavirus dashboard
    # may be no longer work
    print("retrieving data...")
    daily, weekly, cumulative = get_data()

    # save as .csv for convienence
    daily.to_csv('daily_updated.csv')
    weekly.to_csv('weekly_updated.csv')
    cumulative.to_csv('cumulative_updated.csv')

else:
    # use previously retrieved files
    daily = pd.read_csv('./daily.csv').set_index('area_code')
    weekly = pd.read_csv('./weekly.csv').set_index('area_code')
    cumulative = pd.read_csv('./cumulative.csv').set_index('area_code')

# convert dates and sort
weekly['date'] = pd.to_datetime(weekly['date'], dayfirst=True)
weekly.sort_values('date', inplace=True)

daily['date'] = pd.to_datetime(daily['date'], dayfirst=True)
daily.sort_values('date', inplace=True)

cumulative['date'] = pd.to_datetime(cumulative['date'], dayfirst=True)
cumulative.sort_values('date', inplace=True)

# get the council boundary shapefiles (as of Dec 2019) at 500m resolution
geos = gpd.read_file('./shape_files/Local_Authority_Districts_(December_2019)_Boundaries_UK_BUC/Local_Authority_Districts_(December_2019)_Boundaries_UK_BUC.shp')
geos.set_index('lad19cd', inplace=True)
geos = geos[['lad19nm', 'geometry']]
geos = geos.to_crs(epsg=4326)                # important: crs must be set to work with plotly!
geos = geos[geos.index.str.contains("E0")]   # England only

# encode dates as number of days into pandemic (dcc.Slider can't handle datetimes)
def encode_dates(df):
    date_map = {}
    for i, key in enumerate(df.date.astype(str).unique()):
        date_map[key] = i

    df['day'] = df['date'].astype(str).apply(lambda x: date_map[x])
    return df

daily = encode_dates(daily)
weekly = encode_dates(weekly)
cumulative = encode_dates(cumulative)

# join the datasets to the shapefiles
daily = geos.join(daily)
weekly = geos.join(weekly)
cumulative = geos.join(cumulative)

# create a dataframe to update the date-slider
sf = weekly[weekly['cases'].notna()].copy() # slider frame
sf['date'] = pd.to_datetime(sf['date'], dayfirst=True)

# format date-slider for legibility
slider_marks = dict()
month_dict={
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
for month in range(1,13,2):
    for year in range(2020,2022):
        try:
            slider_marks[int(sf.loc[sf['date']==dt.datetime(year,month,1), 'day'].values[0])] = month_dict[month]+str(year)[-2:]

        # date not in dataset
        except IndexError:
            pass


# get limits for date slider
def get_lims(dfs):
    mins, maxs = [], []

    for df in dfs:
        mins.append(df.loc[df['cases'].notna(), 'day'].min())
        maxs.append(df.loc[df['cases'].notna(), 'day'].max())

    start = max(mins)
    end = min(maxs)
    return start, end

slider_min, slider_max = get_lims([daily, weekly, cumulative])


# a convience function for nicer dropdown options
def tidy(x):
    x = re.sub("_", " ", x)
    x = x.title()
    if "Dose" in x:
        x = "Vaccine " + x
    return x


# =============================================================
# Define the app layout
# =============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{
        'name': 'viewport',
        'content': 'width=device-width, initial-scale=1.0'
        }]
    )

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Covid in England"),
            html.Hr(),
            dcc.Dropdown(
                id='council-dropdown',
                options = [{'label': geos.loc[geos.index==x, 'lad19nm'][0], 'value': x} for x in geos.index],
                multi=True,
                value=[],
                placeholder="Select up to 10 councils..."
                ),
            ], xs=12, sm=12, md=12, lg=12, xl=12)
        ]),
   
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='choro-data',
                options=[{'label': tidy(x), 'value': x} for x in ['cases', 'deaths', 'dose_1', 'dose_2']],
                value='cases'
                )            
            ], xs=12, sm=12, md=6, lg=6, xl=6),
        
        dbc.Col([
            dcc.Dropdown(
                id='dataset',
                options=[{'label': tidy(x)+" Figures", 'value': x} for x in ['daily', 'weekly', 'cumulative']],
                value='weekly'
                ),
            ], xs=12, sm=12, md=6, lg=6, xl=6)
        ], justify='between'),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='choropleth'
                ),
            
            dcc.Slider(
                id='date-slider',
                min=slider_min,#weekly['day'].min(),
                max=slider_max,#weekly['day'].max(),
                value=slider_max,#weekly['day'].max(),
                marks=slider_marks
                )
            ],
                xs=12, sm=12, md=6, lg=6, xl=6),
        
        dbc.Col([
            dcc.Graph(
                id='scatter',
                )
            ],
                xs=12, sm=12, md=6, lg=6, xl=6)
        ],
            justify='between', align='center'),
    
    dcc.Store(id='current-selection')
    ])

# =============================================================
# Define the callback - each element can be updated
# by only 1 callback, so for this app it's all-in-one
# =============================================================
@app.callback(
    Output('choropleth', 'figure'),
    Output('council-dropdown', 'value'),
    Output('current-selection', 'data'),
    Output('scatter', 'figure'),
    Input('choropleth', 'figure'),
    Input('choropleth', 'selectedData'),
    State('choropleth', 'relayoutData'),
    Input('council-dropdown', 'value'),
    Input('dataset', 'value'),
    Input('current-selection', 'data'),
    Input('date-slider', 'value'),
    Input('choro-data', 'value')
    )
def update_elements(fig, map_selection, relay, dd_selection, dataset, current_selection, date, metric):
    """
    Update the elements on the app page

      ARGS:
            fig -- the current map
            map_selection -- the highlighted councils
            relay -- current map layout (pan & zoom)
            dd_selection -- current values in the council selection dropdown
            dataset -- the dataframe to use: daily, weekly, or cumulative
            current_selection -- json of coucnil selection from previous callback
            date -- for the choropleth and the date line on the scatterplots
            metric -- to show on the map: cases, deaths, dose_1, dose_2

    RETURN:
            choro -- updated map
            dd_selection -- updated council dropdown selection
            json.sumps(selected_councils) -- for persistence across callbacks
            scatter -- the scatterplots (3 subplots)
        
    """
    

    # who triggered the callback?
    ctx = dash.callback_context
    caller = None if not ctx.triggered else ctx.triggered[0]['prop_id'].split(".")[0]
    selected_councils = []

    def choose_dataset(dataset):
        "choose the DataFrame corresponsing to the `dataset` arg"
        if dataset == "weekly":
            df = weekly

        elif dataset == "daily":
            df = daily

        else:
            df = cumulative

        return df

    # update the choropleth map
    def draw_map(data, date, selection, metric, color_lim):
        "redraw the map when the date, dataset, or metric is changed"
        lat, lon = 52.5617283, -1.468359 # geographic centre of England--a corn field near coventry

        # the basic map layout
        fig = px.choropleth_mapbox(
            data,
            geojson=data.geometry,
            locations=data.index,
            hover_name='area_name',
            opacity=0.75,
            center={'lat': lat, 'lon': lon},
            zoom=5,
            color=metric,
            color_continuous_scale='Reds' if "dose" not in metric else "Greens",
            range_color=(0,color_lim)
           )

        # annotate the selected date on the map
        fig.add_annotation(
            text=data['date'].dt.strftime('%d/%m/%Y').values[0],
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            font={'size': 20}
            )

        # formatting for colorbar ticks
        if ("dose" in metric) and (dataset == "cumulative"):
            suffix, which = "%", "all"

        else:
            suffix, which = "+", "last"


        fig.update_xaxes(fixedrange=True)
        fig.update_layout(
            mapbox_style='carto-positron',
            uirevision="Don't change",     # maintain pan and zoom
            clickmode='event+select',      # select council(s) on (shift)click
            autosize=False,
            coloraxis={                    # colorbar options
                'colorbar_x': 1,
                'colorbar_y': 0.5,
                'colorbar_len': 0.75,
                'colorbar_xpad': 20,
                'colorbar_ypad': 20,
                'colorbar_thickness': 20,
                'colorbar_separatethousands': True,
                'colorbar_title_text': None,
                'colorbar_ticksuffix': suffix,
                'colorbar_showticksuffix': which
                },
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0, 'pad': 4, 'autoexpand': True},
            height=600
            )

        # get DataFrame row indices to highligh coucnils
        # (map is redrawn completely on function call--rehighlight councils)

        if len(selection) == 0:
            # select all coucnils
            points = [data.index.get_loc(i) for i in data.index]

        else:
            # select passed councils
            points = [data.index.get_loc(i) for i in data[data['area_name'].isin(selection)].index]


        # re-highlight the coucnils
        fig.update_traces(selectedpoints=points)

        # don't reset the pan and zoom on redraw
        # https://community.plotly.com/t/how-to-save-current-zoom-and-position-after-filtering/5310
        if relay:
            if 'mapbox.center' in relay:
                fig.update_layout(
                    mapbox_center={'lat': relay['mapbox.center']['lat'], 'lon': relay['mapbox.center']['lon']}
                    )
            if 'mapbox.zoom' in relay:
                fig.update_layout(
                    mapbox_zoom=relay['mapbox.zoom']
                    )
        

        return fig

    
    def draw_scatter(data, date, selection):
        "update the line graphs"

        # choose titles appropriate to the dataset
        if dataset == "daily":
            titles = ['Daily case rate (per 100k)', 'Daily deaths', 'Daily vaccinations (--1st dose, -second dose)']

        elif dataset == "weekly":
            titles = ['Case rate (per 100k), 7-day average', 'Deaths per day, 7-day average', 'Vaccinations, 7-day average (--1st dose, -2nd dose)']

        else:
            titles = ['Cumulative recorded cases', 'Cumulative number of deaths', 'Percentage of population vaccinated (--1st dose, -2nd dose)']
    

        # basic layout
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=titles#["Cases", "Deaths", "Vaccine Doses"]
            )


        fig.update_layout(height=700, width=700)

        # by default, lines are colored by graph, not council
        # get color pallette for consistancy across graphs--limit 10 colors
        cmap = px.colors.qualitative.Plotly

        if len(selection) == 0 :
            # no councils selected: blank graphs
            fig.add_trace(go.Scatter(), row=1, col=1)
            fig.add_trace(go.Scatter(), row=2, col=1)
            fig.add_trace(go.Scatter(), row=3, col=1)

        else:
            # plot the time series for each council
            for i, council in enumerate(selection):
                council_df = data[data['area_name'] == council].copy()
                if len(council_df) != 0:

                    # top: cases
                    fig.add_trace(
                        go.Scatter(
                            x=council_df['date'],
                            y=council_df['cases'],
                            mode='lines',
                            line={'color': cmap[i]},
                            name=council,
                            ),
                        row=1,
                        col=1
                        )

                    # middle: deaths                    
                    fig.add_trace(
                        go.Scatter(
                            x=council_df['date'],
                            y=council_df['deaths'],
                            mode='lines',
                            line={'color': cmap[i]},
                            showlegend=False,
                            name=council
                            ),
                        row=2,
                        col=1
                        )

                    # bottom: dose 1
                    fig.add_trace(
                        go.Scatter(
                            x=council_df['date'],
                            y=council_df['dose_1'],
                            mode='lines',
                            line={'color': cmap[i], 'dash':'dash'},
                            showlegend=False,
                            name=council
                            ),
                        row=3,
                        col=1
                        )

                    # bottom: dose 2                    
                    fig.add_trace(
                        go.Scatter(
                            x=council_df['date'],
                            y=council_df['dose_2'],
                            mode='lines',
                            line={'color': cmap[i]},
                            showlegend=False,
                            name=council
                            ),
                        row=3,
                        col=1
                        )

                    # add a line indicating the date shown on the map
                    vline_x = str(data.loc[data['day']==date, 'date'].values[0])
                    fig.add_vline(x=vline_x, line_width=1, line_dash='dash', line_color='black')


        # "Don't Change" the zoom/pan
        fig.update_layout(uirevision="Don't Change")

        return fig

    # get the dataset (time series) and filter by date (map snapshot)
    df = choose_dataset(dataset)
    dff = df[df['day'] == date]

    # limit for choropleth colorbar across timeseries (avoid high values washing out the plot)
    # heuristic: cap cbar at 95th percentile--may need some refinement
    if "dose" in metric:
        if dataset == "cumulative":
            c_max = 100 # since it's a percentage
            
        else:
            c_max = np.percentile(df['dose_1'].dropna(), 95)

    else:
        c_max = np.percentile(df[metric].dropna(), 95)


    # initialization
    if caller is None:
        choro = draw_map(dff, date, selected_councils, metric, color_lim=c_max)

    # add/remove council(s) from selection
    elif caller == "choropleth":
        choro=go.Figure(fig)
        selected_councils = []
        if map_selection is not None:
            for point in map_selection['points']:
                council = point['hovertext']
                selected_councils.append(council)

            points = [dff.index.get_loc(i) for i in dff[dff['area_name'].isin(selected_councils)].index]
            dd_selection = dff.iloc[points].index

        else:
            #select all coucnils
            points = [dff.index.get_loc(i) for i in dff.index]
            dd_selection = []

        # perform the update
        choro.update_traces(selectedpoints=points)

        
    # add/remove council(s) from selection
    elif caller == "council-dropdown":
        choro=go.Figure(fig)
        selected_councils = []
        selected_councils_code = []
        if dd_selection != []:
            for council_code in dd_selection:
                selected_councils_code.append(council_code)

            points = [dff.index.get_loc(i) for i in selected_councils_code]

        else:
            # select all councils
            points = [dff.index.get_loc(i) for i in dff.index]

        # perform the update
        selected_councils = dff.loc[dff.index.isin(selected_councils_code),'area_name'].values.tolist()
        choro.update_traces(selectedpoints=points)


    # update the date, dateset, or metric shown on the map
    elif (caller == "date-slider") or (caller == "dataset") or (caller == "choro-data"):
        #maintain current selection
        selected_councils = json.loads(current_selection)

        # update the map & keep the dropdown selection
        choro = draw_map(dff, date, selected_councils, metric, color_lim=c_max)
        dd_selection = dff[dff['area_name'].isin(selected_councils)].index

    else:
        # if you're here then something went drastically wrong...
        pass

    # update the scatter plots
    scatter = draw_scatter(df, date, selected_councils)

    return choro, dd_selection, json.dumps(selected_councils), scatter


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True)

