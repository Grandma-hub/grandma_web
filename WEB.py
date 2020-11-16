import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import base64
import re
from facebook_scraper import get_posts
import joblib
from numpy import mean

# external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

logoImage = 'pp.png'
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
                "filter": "progid:DXImageTransform.Microsoft.gradient(startColorstr='#46fcb1',endColorstr='#3ffb6e',GradientType=1)"
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


def predictor(document):
    document_vectorized = vectorizer.transform(document)
    return mean(model.predict(document_vectorized))


@app.callback(Output('result', 'children'),
              [Input('check-button', 'n_clicks')],
              [State('input-url', 'value')]
              )
def update_output(n_clicks, content):  # Displayes the image
    """Displays an inputted image on the page."""
    content = re.sub("(?:https?:\/\/)?(?:www\.)?facebook\.com\/?(?:\/)",
                     "", str(content))
    list_of_content = []
    for post in get_posts('{}'.format(content), pages=20):
        if post["text"] != None:
            list_of_content.append(post["text"])

    result = predictor(list_of_content)
    return "Percentage of hate-speech: {}%".format(result)


if __name__ == '__main__':
    app.run_server(debug=True)
