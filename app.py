import pandas                   as pd
import matplotlib.pyplot        as plt
import seaborn                  as sns
import numpy                    as np
import math
from datetime import datetime   as dt
from datetime import timedelta
import dash
#from dash import html
#from dash import dcc
import dash_html_components             as html
import dash_core_components             as dcc
import dash_bootstrap_components        as dbc
import plotly.express                   as px
import plotly.graph_objs                as go
from dash.dependencies import Input, Output
from http import server
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import statsmodels
#%matplotlib inline
import ssl
import requests
from urllib.request import urlopen
import joblib


#####################################################################################
#####################################################################################
######################          Read datasets


s = pd.read_csv('data/suesca.csv')
t = pd.read_csv('data/tocancipa.csv')

#with urlopen('https://farmappdata.blob.core.windows.net/data/suesca.csv') as response:
#    s = pd.read_csv(response)
#with urlopen('https://farmappdata.blob.core.windows.net/data/tocancipa.csv') as response:
#    t = pd.read_csv(response)

t = t.rename(columns={'sensacion_térmica': 'sensacion_termica'})
t['ciudad'] = 'tocancipa'
t.PLAGA = t.PLAGA.str.capitalize()

s['ciudad'] = 'suesca'
s.PLAGA = s.PLAGA.replace('ARAÇ?A', 'Araña')
s.PLAGA = s.PLAGA.str.capitalize()

s2 = s.sample(n=50000)
t2 = t.sample(n=50000)

data = t2.append(s2)

cities = ['Suesca', 'Tocancipá']


#####################################################################################
#####################################################################################
#############################           K-nearest neighbors (kNN) model

########### Data to train K-nearest neighbors (kNN) model
knn_data= t.drop(columns=['Unnamed: 0','datetime','PLAGA','LAT','LON','PRODUCTO','NAVE','CAMA','BLOQUE',
                                   'CUADRO','temperatura_interna','humedad_interna','sensacion_termica','radiacion_solar'])
knn_data=knn_data.dropna()

X = knn_data.drop(columns = ['SEVERIDAD'])
y = knn_data.SEVERIDAD

# Select and extract numerical cols
numerical_cols = X.select_dtypes(include='number').columns
X_num = X[numerical_cols]

# Scale numeric features
scaler = MinMaxScaler()

X_ready = pd.DataFrame(scaler.fit_transform(X_num))

# Scaler removes the column names, so put them back
X_ready.columns = X_num.columns
X_ready.head()


#creating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X_ready, y.astype('int'), 
                                        test_size=.4, random_state=1234, shuffle=True)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy_score(y_test, predictions)


########### Function to clean IDEAM weather forecast data
def prepro(ideam, forward_days):
  #ideam= pd.read_csv(x, encoding='latin-1')
  limpio= ideam.drop(columns=['Cod_Div','Latitud','Longitud','Región','Departamento','Dirección del Viento','Cobertura total nubosa','Probabilidad de Tormenta'])
  limpio = limpio.reindex(columns=['Fecha','Hora','Temperatura','Humedad','Presión','Punto de Rocío','Velocidad del Viento','Precipitación (mm/h)'])
  limpio['dia']=pd.to_datetime(limpio['Fecha'], format="%Y/%m/%d").dt.day

  #current_day = dt.now().date().day
  current_day = limpio['dia'][0]
  forecast_day = forward_days + current_day
  limpio=limpio.loc[limpio['dia'] == forecast_day]

  limpio['Humedad']=pd.to_numeric(limpio['Humedad'],errors = 'coerce')
  X=limpio.groupby(by=["dia"]).mean()
  Y=limpio.groupby(by=["dia"]).sum()
  X['Precipitación (mm/h)']=Y['Precipitación (mm/h)']
  df = X.rename(columns={'Temperatura':'temperatura_externa','Humedad':'humedad_externa', 'Presión':'presion','Punto de Rocío':'punto_de_rocio', 'Velocidad del Viento':'velocidad_del_viento','Precipitación (mm/h)':'lluvia_diaria'})
  return df

########## Classification function
def det_plagas(x):
  y_pred=knn.predict(x)
  
  x=x.reset_index()
  Acaros=[]
  Afidos=[]
  Botrytis=[]
  MildeoPolvoso=[]
  MildeoVelloso=[]
  Minador=[]
  Trips=[]

  for i in range(len(x)):
    for p in x.index:
        
        if ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=14.0) and (x['humedad_externa'][p] >= 85.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            Acaros.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 15.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 56.0 and x['humedad_externa'][p] <=85.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            Acaros.append('Medio')
        elif ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 35.0 and x['humedad_externa'][p] <=55.0) and (x['lluvia_diaria'][p] == 0.0)):
            Acaros.append('Alto')
        
        if ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=14.0) and (x['humedad_externa'][p] >= 30.0 and x['humedad_externa'][p] <=44.0) and (x['lluvia_diaria'][p] == 0.0)):
            Afidos.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 15.0 and x['temperatura_externa'][p] <=19.0) and (x['humedad_externa'][p] >= 45.0 and x['humedad_externa'][p] <=59.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            Afidos.append('Medio')
        elif ((x['temperatura_externa'][p] >= 20.0 and x['temperatura_externa'][p] <=26.0) and (x['humedad_externa'][p] >= 60.0 and x['humedad_externa'][p] <=80.0) and (x['lluvia_diaria'][p] > 10.0)):
            Afidos.append('Alto')
            
        if ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 38.0 and x['humedad_externa'][p] <=64.0) and (x['lluvia_diaria'][p] == 0.0)):
            Botrytis.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 16.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 65.0 and x['humedad_externa'][p] <=94.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            Botrytis.append('Medio')
        elif ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=15.0) and (x['humedad_externa'][p] >= 95.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            Botrytis.append('Alto')
        
        if ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=14.0) and (x['humedad_externa'][p] >= 75.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            MildeoPolvoso.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 15.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 51.0 and x['humedad_externa'][p] <=75.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            MildeoPolvoso.append('Medio')
        elif ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 35.0 and x['humedad_externa'][p] <=50.0) and (x['lluvia_diaria'][p] == 0.0)):
            MildeoPolvoso.append('Alto')
            
        if ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 38.0 and x['humedad_externa'][p] <=64.0) and (x['lluvia_diaria'][p] == 0.0)):
            MildeoVelloso.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 16.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 65.0 and x['humedad_externa'][p] <=94.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            MildeoVelloso.append('Medio')
        elif ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=15.0) and (x['humedad_externa'][p] >= 95.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            MildeoVelloso.append('Alto')
            
        if ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 38.0 and x['humedad_externa'][p] <=64.0) and (x['lluvia_diaria'][p] == 0.0)):
            Minador.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 16.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 65.0 and x['humedad_externa'][p] <=94.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            Minador.append('Medio')
        elif ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=15.0) and (x['humedad_externa'][p] >= 95.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            Minador.append('Alto')
        
        if ((x['temperatura_externa'][p] >= 0.0 and x['temperatura_externa'][p] <=14.0) and (x['humedad_externa'][p] >= 85.0 and x['humedad_externa'][p] <=100.0) and (x['lluvia_diaria'][p] > 10.0)):
            Trips.append('Bajo')
        elif ((x['temperatura_externa'][p] >= 15.0 and x['temperatura_externa'][p] <=24.0) and (x['humedad_externa'][p] >= 56.0 and x['humedad_externa'][p] <=85.0) and (x['lluvia_diaria'][p] >= 1.0  and x['lluvia_diaria'][p] <=10.0)):
            Trips.append('Medio')
        elif ((x['temperatura_externa'][p] >= 25.0 and x['temperatura_externa'][p] <=35.0) and (x['humedad_externa'][p] >= 35.0 and x['humedad_externa'][p] <=55.0) and (x['lluvia_diaria'][p] == 0.0)):
            Trips.append('Alto')
        
        d = {'Acaros': Acaros, 'Afidos':Afidos,'Botrytis':Botrytis,'MildeoPolvoso':MildeoPolvoso,'MildeoVelloso':MildeoVelloso,'Minador':Minador,'Trips':Trips}
        acc=accuracy_score(y_test, predictions)
  
  return d, y_pred, acc





#####################################################################################
#####################################################################################
#############################           Quick settings to use in layout
def label_variable(variable):
    labels_variable = {
    'temperatura_externa': 'Temperatura Externa',
    'temperatura_interna': 'Temperatura Interna',
    'humedad_externa': 'Humedad Externa',
    'humedad_interna': 'Humedad Interna',
    'presion': 'Presión',
    'punto_de_rocio': 'Punto de Rocío',
    'sensacion_termica': 'Sensación Térmica',
    'velocidad_del_viento': 'Velocidad del Viento',
    'lluvia_diaria': 'Lluvia Diaria',
    'radiacion_solar': 'Radiacion Solar'
    }
    return labels_variable[variable]


def select_data(city_selected):
    if city_selected is None or len(city_selected) == 0:
        return data
    elif city_selected == 'Tocancipá':
        return t2
    else:
        return s2


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None, height=200)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

br = html.Br()
Mbr = html.Div([br, br, br])
l = html.Hr()



#####################################################################################
#####################################################################################
#############################           Setup the app 
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            'name': 'viewport', 
            'content': 'width=device-width, initial-scale=1.0'
        }
    ],
    suppress_callback_exceptions=True
)
app.title = "FarmApp - Plagues Prediction"
server = app.server



tarjeta_alerta = dbc.Card(
                    [
                        dbc.CardHeader(
                                html.H4(
                                    "Sistema de Alerta", 
                                    className="card-title",
                                    style={
                                        'textAlign': 'center'
                                    }                                                                   
                                ),
                                style={
                                    'background': '#F5F9EA'
                                }
                        ),
                        dbc.CardBody(
                                html.P(
                                    '''
                                    El pronóstico para el producto XXX ubicada en la
                                    lat XX, lon XX del bloque XXX cuenta aparentemente
                                    con una temperatura XX°C, sensación térmica XX, lluvia XXmm,
                                    radiación solar media XX y humedad media de XX, puede verse
                                    afectada por la plaga XXX.

                                    Comuníquese con el administrador para recibir asesoría sobre el cultivo.
                                    ''',
                                    id='card_alert_body',
                                    className="card-text"
                                ),
                        ),
                        dbc.CardFooter(
                            html.P(
                                    '', 
                                    id='severity_level_alert',
                                    style={
                                        'textAlign': 'center'
                                    }                                                                   
                                ),
                            id='card_alert_footer',
                            #style={
                            #    'background': '#FF5757'
                            #}
                        ),
                    ]
                )




#####################################################################################
#####################################################################################
#############################           Layout Dash App
app.layout = html.Div(
    children=[
        dbc.Row(
            [   
                
                #####################################################################################
                #############################           Sidebar panel: control panel
                html.Div(
                    children=[
                        #farmapp logo image
                        html.Div(
                            html.Img(
                                className='logo',
                                src= 'https://i.ibb.co/cxP34bY/farmapp.png',
                                style={
                                    'height':'140px',
                                    #'margin':'0px 50px'
                                        }
                            ),
                            style={'textAlign': 'center'}
                        ),
                        l,
                        html.H3('Control Panel'),
                        dcc.Dropdown(
                            id='cities_dropdown',
                            options= [
                                {
                                    'label': str(i),
                                    'value': str(i)
                                }
                                for i in cities
                            ],
                            multi=False,
                            placeholder='City',
                            style={
                                "width": "20rem", 
                                #'display': 'inline-block'
                            } 
                        ),
                        
                        br,
                        dcc.Dropdown(
                            id='plagues_dropdown',
                            options= [],
                            multi=True,
                            placeholder='Plague',
                            style={
                                "width": "20rem", 
                                #'display': 'inline-block'
                            }            
                        ),

                        br,
                        dcc.Dropdown(
                            id='variables_dropdown',
                            options= [],
                            multi=False,
                            placeholder='Variable',
                            style={
                                "width": "20rem", 
                                #'display': 'inline-block'
                            }            
                        ),

                        br,
                        dcc.Dropdown(
                            id='variables2_dropdown',
                            options= [],
                            multi=False,
                            placeholder='Scatter Plot',
                            style={
                                "width": "20rem", 
                                #'display': 'inline-block'
                            }            
                        ),
                        l,
                        html.H5('IDEAM Weather Forecast'),
                        dbc.Row(
                            [   
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id='variables_ideam_dropdown',
                                            options= [],
                                            multi=False,
                                            placeholder='Select Variable',
                                                        
                                        )

                                    ],
                                )
                            ]
                        ),
                        
                        br,
                        html.H5('Forecasting Date'),                        
                        dcc.Slider(
                            id='days_slider',
                            min=0,
                            max=3,
                            step=1,
                            value=2,
                            #marks=None,
                            #tooltip={
                            #    "placement": "bottom", 
                            #    "always_visible": True
                            #    }
                        ),
                        html.P(
                            children=[],
                            id='forecast_date',
                            style={
                                'textAlign': 'center',
                                'color': '#55672B'
                            }
                        ),
                        br,
                        html.Div(
                            html.Img(
                                className='logo',
                                src= 'https://i.ibb.co/3TLjFcm/ager-logo.png',
                                style={
                                    'height':'150px',
                                }
                            ), 
                            style={'textAlign': 'center'},
                        ),
                        html.P(
                            'by Team 36',
                            style={
                                'textAlign': 'center',
                                'color': '#09642D',
                                'fontSize': 12
                            }
                        ),
                    ],
                    style={
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "bottom": 0,
                        "width": "22rem",
                        "margin-left": "1rem",
                        "margin-top": "1rem",
                        "padding": "1rem 1rem",
                        "background-color": "#F5F9EA",
                    }
                ),

                #####################################################################################
                #############################           Content Dash Plots and stats
                html.Div(
                    children=[

                        #Header
                        html.Div(
                            children=[
                                html.H1(
                                    'Pest Control and Prediction in Flower Crops',
                                    style={
                                        'textAlign': 'center',
                                        'color': '#55672B'
                                    }
                                )
                            ],
                            style={
                                #"position": "fixed",
                                #"margin-left": "24.5rem",
                                #"margin-right": "1rem",
                                #"margin-top": "1rem",
                                'left': 0,
                                #"top": 0,
                                'right':0,
                                "padding": "1rem 1rem",
                                "background-color": "#C8EB7E"
                            }
                        ),
                        br,

                        #####################################################################################
                        #############################           Plots and stats
                        html.Div(
                            [
                                dbc.Row(
                                    [   
                                        dbc.Col(
                                            [   
                                                dbc.Row(
                                                    dbc.Col(
                                                        html.H5(
                                                            'Gráficos Pronóstico Variables IDEAM',
                                                            style={
                                                                'textAlign': 'center',
                                                                'color': '#55672B'
                                                            }
                                                        )
                                                     )
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dcc.Graph(id='box_variables_ideam', figure = blank_fig())),
                                                        dbc.Col(dcc.Graph(id='prob_density_variables_ideam', figure = blank_fig()))
                                                    ]
                                                )
                                            ]

                                        ),                                        
                                        dbc.Col(tarjeta_alerta)
                                    ]
                                ),
                                l,
                                br,
                                dbc.Row(
                                    [
                                        dbc.Col(dcc.Graph(id='mapa_1')),
                                        dbc.Col(dcc.Graph(id='bar_plagues')),
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(dcc.Graph(id='box_variables')),
                                        dbc.Col(dcc.Graph(id='prob_density_variables')),
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(dcc.Graph(id='scatterplot'))
                                    ]
                                ),










                                #footer here
                                


                            ]

                        )
                                
                                
                        
                        
                    ],
                    #style sidebar panel (control panel)
                    style={
                        "position": "fixed",
                        "margin-left": "23.5rem",
                        "margin-right": "1rem",
                        "margin-top": "1rem",
                        'left': 0,
                        'top': 0,
                        'right':0,
                        'bottom':0,
                        #"padding": "1rem 1rem",
                        #"background-color": "#F5F9EA",
                        "overflow": "scroll"
                    }  
                )
            ]

        )

    ]
)
        







##################################################################################
##################    Callbacks    ##################
##################################################################################



#enable IDEAM forecast button
@app.callback(    
    Output('days_slider', 'disabled'),
    Input('cities_dropdown', 'value')
)
def enable_slider_forecast_day(ciudad):
    if ciudad is None or len(ciudad) == 0:
        return True
    else:
        return False


#update IDEAM dropdown variables
@app.callback(    
    [Output('variables_ideam_dropdown', 'disabled'),
    Output('variables_ideam_dropdown', 'options'),
    Output('variables_ideam_dropdown', 'value')],
    Input('cities_dropdown', 'value')
)
def update_var_ideam(ciudad):
    if ciudad is None or len(ciudad) == 0:
        deshabilitado = True
    else:
        deshabilitado = False
        
    var_ideam = ['Temperatura', 'Humedad', 'Presión', 'Precipitación (mm/h)', 'Velocidad del Viento', 'Punto de Rocío']
    return deshabilitado, [{'label': str(i), 'value': i} for i in var_ideam], ''



@app.callback(    
    [Output('plagues_dropdown', 'options'),
    Output('plagues_dropdown', 'value')],
    Input('cities_dropdown', 'value')
)
def update_plagues_dropdown(ciudad):
    data_plagues = select_data(ciudad)

    sorted_plagues = data_plagues.PLAGA.sort_values().unique().tolist()
    return [{'label': str(i), 'value': i} for i in sorted_plagues], ''


#update variables from Farmapp dropdown and then plotting (univariable anaylsis)
@app.callback(    
    [Output('variables_dropdown', 'options'),
    Output('variables_dropdown', 'value')],
    Input('cities_dropdown', 'value')
)
def update_variables_dropdown(ciudad):
    data_variables = select_data(ciudad)
    variables = data_variables.columns.tolist()[11:-1]

    return [{'label': str(i), 'value': i} for i in variables], ''


#update 2nd variables dropdown and then plot the scatter (bivariable anaylsis)
@app.callback(    
    [Output('variables2_dropdown', 'options'),
    Output('variables2_dropdown', 'value')],
    Input('cities_dropdown', 'value')
)
def update_variables2_dropdown(ciudad):
    data_variables = select_data(ciudad)

    variables = data_variables.columns.tolist()[11:-1]
    return [{'label': str(i), 'value': i} for i in variables], ''


#Render Map cities and crops data FarmApp
@app.callback(
    Output('mapa_1', 'figure'),
    [Input('cities_dropdown', 'value'),
    Input('plagues_dropdown', 'value')]
)
def mapa(ciudad, plaga):
    if ciudad is None or len(ciudad) == 0:
        data_fig = data
        zoom = 10.9
    elif ciudad == 'Tocancipá':
        data_fig = t2
        zoom = 14.5
    else:
        data_fig = s2
        zoom = 12
    
    if plaga is None or len(plaga) == 0:
        data_fig = data_fig
    else:
        data_fig = data_fig.loc[data_fig.PLAGA.isin(plaga)]


    fig = px.scatter_mapbox(data_fig,
                        lat="LAT",
                        lon="LON",
                        #hover_name= columna,
                        #hover_data=lista,
                        color='ciudad',
                        height=400,
                        size_max=0,
                        zoom=zoom,
                        color_discrete_sequence=[px.colors.qualitative.Dark2[0],
                                                px.colors.qualitative.D3[1]],
                        #center = {'lat': data_fig.LAT.mean(), 'lon': data_fig.LON.mean()}
                        )
    fig.update_layout(mapbox_style="open-street-map") #estilo de mapa
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0)) #quitan las margenes = 0
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.98,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=13,
                color="black"
            ),
        )
    )
    return fig


#Render plot plagues in cities
@app.callback(
    Output('bar_plagues', 'figure'),
    [Input('cities_dropdown', 'value'),
    Input('plagues_dropdown', 'value')]
)
def barplot_plagues(ciudad, plagas):
    data_fig = select_data(ciudad)

    data_fig = data_fig.groupby(['ciudad','PLAGA'])[['SEVERIDAD']].count().reset_index().rename(columns={'SEVERIDAD': 'Cantidad'}).sort_values('Cantidad', ascending=False)
    if plagas is None or len(plagas) == 0:
        data_fig = data_fig
    else:
        data_fig = data_fig.loc[data_fig.PLAGA.isin(plagas)]
    
    fig = px.histogram(data_fig.head(10), x="PLAGA", y="Cantidad",
                    color='ciudad', barmode='group',
                    height=400,
                    template='plotly_white',
                    color_discrete_sequence=px.colors.qualitative.Pastel2,
                    labels={
                        "PLAGA": "Tipo de plaga",
                        "sum of Cantidad": "Cantidad",
                        "ciudad": "Ciudad"
                    }
                )
    fig.update_layout(
        title={
            'text': 'Plagas por Ciudad',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig


#Render boxplot variables Farmapp
@app.callback(
    Output('box_variables', 'figure'),
    [Input('cities_dropdown', 'value'),
    Input('plagues_dropdown', 'value'),
    Input('variables_dropdown', 'value')]
)
def boxplot_var(ciudad, plagas, variable):
    data_fig = select_data(ciudad)
    
    if plagas is None or len(plagas) == 0:
        data_fig = data_fig
        color = 'ciudad'
        legend=False
    else:
        data_fig = data_fig.loc[data_fig.PLAGA.isin(plagas)]
        color='PLAGA'
        legend=True
    
    if variable is None or len(variable) == 0:
        fig = px.box(
            data_fig, x= 'ciudad', 
            y='temperatura_interna', 
            color=color, template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel2,
            labels={
                "ciudad": "Ciudad",
                'temperatura_interna': label_variable('temperatura_interna'),
                "PLAGA": "Tipo de Plaga"
            }
        )
        title = label_variable('temperatura_interna')
    else:
        fig = px.box(
            data_fig, x= 'ciudad', 
            y=variable, 
            color=color, template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel2,
            labels={
                "ciudad": "Ciudad",
                variable: label_variable(variable),
                "PLAGA": "Tipo de Plaga"
            }
        )
        title = label_variable(variable)
    
    fig.update_layout(
        showlegend=legend, 
        margin=dict(b=0, l=0, r=0),
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig



#Render density probability plot variables Farmapp
@app.callback(
    Output('prob_density_variables', 'figure'),
    [Input('cities_dropdown', 'value'),
    Input('variables_dropdown', 'value')]
)
def density_var(ciudad, variable):
    data_fig = select_data(ciudad)

    if variable is None or len(variable) == 0:
        var = 'temperatura_interna'
    else:
        var = variable

    fig = px.histogram(data_fig, x=var, 
                histnorm='probability density', 
                color='ciudad',
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Pastel2,
                labels={
                    var: label_variable(var),
                    "ciudad": "Ciudad"
                }
                #title= f'Densidad de Probabilidad de {label_variable(var)}'
        )
    
    fig.update_layout(
        showlegend=True, 
        margin=dict(b=0, l=0, r=0),
        title={
            'text': f'Densidad de Probabilidad de {label_variable(var)}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig


#Render Scatter plot
@app.callback(
    Output('scatterplot', 'figure'),
    [Input('cities_dropdown', 'value'),
    Input('variables_dropdown', 'value'),
    Input('variables2_dropdown', 'value')]
)
def scatter_vars(ciudad, variable1, variable2):
    data_fig = select_data(ciudad)

    if variable1 is None or len(variable1) == 0:
        var1 = 'temperatura_interna'
    else:
        var1 = variable1

    if variable2 is None or len(variable2) == 0:
        var2 = var1
    else:
        var2 = variable2

    fig = px.scatter(data_fig, x=var1, y=var2, #color='ciudad',
                #trendline="ols",
                template='plotly_white',
                color_discrete_sequence=[px.colors.qualitative.Pastel2[0]],
                labels={
                    var1: label_variable(var1),
                    var2: label_variable(var2),
                    "ciudad": "Ciudad"
                }
                #title= f'Densidad de Probabilidad de {label_variable(var)}'
        )
    
    fig.update_layout(
        showlegend=True, 
        margin=dict(b=0, l=0, r=0),
        title={
            'text': f'Gráfico de Dispersión {label_variable(var1)} vs {label_variable(var2)}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig


#Render plots (boxplot and probability density) from ideam variables
@app.callback(
    [Output('box_variables_ideam', 'figure'),
    Output('prob_density_variables_ideam', 'figure')],
    [Input('cities_dropdown', 'value'),
    Input('variables_ideam_dropdown', 'value')]
)
def fig_var_ideam(ciudad, var_ideam):

    if ciudad is None or len(ciudad) == 0 or var_ideam is None or len(var_ideam) == 0:
        return blank_fig(), blank_fig()

    elif ciudad == 'Tocancipá':
        #ideam tocancipa
        url_toc = 'http://mipronostico.ideam.gov.co/IdeamWebApp2/Ideam/getDatosAbiertos/pronosticos/ws/generador.php?cod=25817000&dato=csv&tipo=prono'
        ideam_data_toc = pd.read_csv(url_toc, encoding='latin-1')

        if var_ideam == 'Temperatura' or var_ideam == 'Humedad':
            text = 'Externa'
        else:
            text = ''

        fig = px.box(
            ideam_data_toc,  
            y=var_ideam, 
            color_discrete_sequence=[px.colors.qualitative.Set2[4]],
            template='plotly_white',
            height=200
        )
        fig.update_layout(
            margin=dict(b=0, l=0, r=0),
            title={
                'text': f'{var_ideam} {text} en Tocancipá',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )


        fig2 = px.histogram(ideam_data_toc, x=var_ideam, 
                histnorm='probability density',
                color_discrete_sequence=[px.colors.qualitative.Pastel2[0]],
                template='plotly_white',
                height=200
        )
        fig2.update_layout(
            margin=dict(b=0, l=0, r=0),
            title={
                'text': f'Densidad de probabilidad de <br> {var_ideam} {text} en Tocancipá',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )


        return fig, fig2

    else:
        #ideam suesca
        url_sues = 'http://mipronostico.ideam.gov.co/IdeamWebApp2/Ideam/getDatosAbiertos/pronosticos/ws/generador.php?cod=25772000&dato=csv&tipo=prono'
        ideam_data_sues = pd.read_csv(url_sues, encoding='latin-1')

        if var_ideam == 'Temperatura' or var_ideam == 'Humedad':
            text = 'Externa'
        else:
            text = ''

        fig = px.box(
            ideam_data_sues,  
            y=var_ideam, 
            color_discrete_sequence=[px.colors.qualitative.Set2[4]],
            template='plotly_white',
            height=200
        )

        fig.update_layout(
            margin=dict(b=0, l=0, r=0),
            title={
                'text': f'{var_ideam} {text} en Suesca',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

        fig2 = px.histogram(ideam_data_sues, x=var_ideam, 
                histnorm='probability density',
                color_discrete_sequence=[px.colors.qualitative.Pastel2[0]],
                template='plotly_white',
                height=200
        )
        fig2.update_layout(
            margin=dict(b=0, l=0, r=0),
            title={
                'text': f'Densidad de probabilidad de <br> {var_ideam} {text} en Suesca',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        return fig, fig2


#Update forescating date selected
@app.callback(
    Output('forecast_date', 'children'),
    Input('days_slider', 'value')
)
def update_label_forecasting_date(day):
    #today = dt.now()
    url_ideam_toca = 'http://mipronostico.ideam.gov.co/IdeamWebApp2/Ideam/getDatosAbiertos/pronosticos/ws/generador.php?cod=25817000&dato=csv&tipo=prono'
    ideam_toca= pd.read_csv(url_ideam_toca, encoding='latin-1')
    ideam_toca['Fecha']=pd.to_datetime(ideam_toca['Fecha'], format="%Y/%m/%d")
    today = ideam_toca['Fecha'][0]
    forecasting_date = str((today + timedelta(days=day)).date())
    return forecasting_date



@app.callback(
    [Output('card_alert_body', 'children'),
    Output('card_alert_footer', 'style'),
    Output('severity_level_alert', 'children')],
    [Input('cities_dropdown', 'value'),
    Input('days_slider', 'value')]
)
def forecasting(ciudad, day):
    if ciudad is None or len(ciudad) == 0:
        mensaje = 'Seleccioné una ciudad para realizar la predicción'
        alert_color = '#FAFAFA'
        sl = ''

    elif ciudad == 'Tocancipá':
        url_ideam_toca = 'http://mipronostico.ideam.gov.co/IdeamWebApp2/Ideam/getDatosAbiertos/pronosticos/ws/generador.php?cod=25817000&dato=csv&tipo=prono'
        ideam_toca= pd.read_csv(url_ideam_toca, encoding='latin-1')
        results = det_plagas(prepro(ideam_toca, day))
        ideam_toca['Fecha']=pd.to_datetime(ideam_toca['Fecha'], format="%Y/%m/%d")
        today = ideam_toca['Fecha'][0]
        forecasting_date = str((today + timedelta(days=day)).date())

        if results[1][0] == 3:
            severidad = 'high'
            alert_color = '#FADBD8'
            sl = 'Severity 3 in crops'
        elif results[1][0] == 2:
            severidad = 'medium'
            alert_color = '#FDEBD0'
            sl = 'Severity 2 in crops'
        else:
            severidad = 'low'
            alert_color = '#D5F5E3'
            sl = 'Severity 1 in crops'

        mensaje = f'''By {forecasting_date} with the current IDEAM weather forecast, 
                    the severity of the pest in Tocancipá crops is expected to be {severidad}: level {results[1][0]}'''

    elif ciudad == 'Suesca':
        url_ideam_sues = 'http://mipronostico.ideam.gov.co/IdeamWebApp2/Ideam/getDatosAbiertos/pronosticos/ws/generador.php?cod=25772000&dato=csv&tipo=prono'
        ideam_sues= pd.read_csv(url_ideam_sues, encoding='latin-1')
        results = det_plagas(prepro(ideam_sues, day))
        ideam_sues['Fecha']=pd.to_datetime(ideam_sues['Fecha'], format="%Y/%m/%d")
        today = ideam_sues['Fecha'][0]
        forecasting_date = str((today + timedelta(days=day)).date())

        if results[1][0] == 3:
            severidad = 'high'
            alert_color = '#FADBD8'
            sl = 'Severity 3 in crops'
        elif results[1][0] == 2:
            severidad = 'medium'
            alert_color = '#FDEBD0'
            sl = 'Severity 2 in crops'
        else:
            severidad = 'low'
            alert_color = '#D5F5E3'
            sl = 'Severity 1 in crops'

        mensaje = f'''By {forecasting_date} with the current IDEAM weather forecast, 
                    the severity of the pest in Suesca crops is expected to be {severidad}: level {results[1][0]}'''

    else:
        mensaje = 'La ciudad ingresada no es válida'
    
    return mensaje, {'background': alert_color}, sl



####=============Run
if __name__ == '__main__':
    app.run_server(
        #host='0.0.0.0', 
        port='8080', 
        debug=True
    )