import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import base64
import re
import joblib
from numpy import mean
import pandas as pd
import requests
import psycopg2
from config import config
# from boto.s3.connection import S3Connection


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = 'Grandma' 
server = app.server

# api_key_scrape = S3Connection(os.environ['api_scraper'])
# #api scraper
# endpoint = "https://extractorapi.com/api/v1/extractor"

# external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

params = config()

logoImage = 'grandma_icon.png'
encoded_image = base64.b64encode(open(logoImage, 'rb').read())
model = joblib.load("model_hs.sav")
vectorizer = joblib.load("vectorizer.sav")

app.layout = html.Div(children=[
    html.Img(
        src='data:image/png;base64,{}'.format(encoded_image.decode()),
        width="10%",
        style={
            'display': 'inline-block'
        }),

    html.Div([
        dcc.Input(
            id='input-url',
            type="url",
            placeholder="input the URL",
            style={
                'width': '44%',
                'height': '25px',
                'lineHeight': '60p',
                'borderStyle': 'hidden',
                'borderRadius': '15px',
                'text-align': 'center',
                "margin-left": "390px",
                "margin-top": "250px",
                "margin-right": "auto",
                "transform": "scale(1.5)",
                "background": "white",
            },
            multiple=False
        ),
        dbc.Button("Check", id='check-button', color="light", n_clicks=0, style={"margin-left": "280px",
                                                                                 "height": "40px",
                                                                                 "transform": "scale(1.5)"}),
        dcc.Loading(
            id="loading",
            type="cube",
            fullscreen=True,
            children=html.Div(id="result",
                              children="",
                              style={"text-align": "center",
                                     "padding": "20px",
                                     'display': 'inline-block',
                                     "height": "auto",
                                     "vertical-align": "top",
                                     "word-wrap": "break-word",
                                     "fontSize": "40px",
                                     "color": "white"}),
            style={"background": "black"}
        ),

    ])

])


def insert_db(website, perc):
    """ insert multiple vendors into the vendors table  """
    conn = None
    try:
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        postgres_insert_query = """ INSERT INTO website_check (website, hate_speech) VALUES (%s,%s)"""
        record_to_insert = (website, perc)
        cur.execute(postgres_insert_query, record_to_insert)
        # execute the INSERT statement
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def predictor(document):
    document_vectorized = vectorizer.transform(document)
    return mean(model.predict(document_vectorized))


@app.callback(Output('result', 'children'),
              [Input('check-button', 'n_clicks')],
              [State('input-url', 'value')]
              )
def update_output(n_clicks, content):  # Displayes the image
    """Displays an inputted image on the page."""

    params = {
      "apikey": api_key_scrape,
      "url": content
    }


    # connect to the PostgreSQL database
    conn = psycopg2.connect(**params)
    
    df = pd.read_sql_query('select * from website_check',con=conn)
    exist = content in list(df["website"])
    
    content_text = list(requests.get(endpoint, params=params).json()["text"])

    
    
    if not exist:
        results = predictor(content_text)
        result= float(results)*100
        insert_db(content, str(result))
    else:
        result = float(list(df.loc[df['website'] == content, 'hate_speech'])[0])
      
    conn.close()
    return "Percentage of hate-speech: {}%".format(int(result))


if __name__ == '__main__':
    app.run_server(debug=False)
