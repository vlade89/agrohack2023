import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)


DEFAULT_TSNE = 40

CLUSTER_VECTOR_SIZE = 16
NODE_SIZE = 20
INIT_N_CLUSTERS = 8
SCALE_PARAM = 4000

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

network_df = pd.read_csv("clustered_data.csv", index_col=0)


def prepare_coors(raw: str, goal_size=CLUSTER_VECTOR_SIZE):
    buf = raw[1:-1].replace("\n", " ").split(" ")
    try:
        res = tuple(float(i.strip()) for i in buf if len(i) > 0)
        res_to_dict = {i: r for i, r in enumerate(res)}
        cols = list(res_to_dict.keys())
    except Exception as e:
        print(e)
        raise e
    return res_to_dict


coords = pd.DataFrame.from_records(
    network_df["ft_vectors"].map(lambda x: prepare_coors(x)).values
)
for col in range(CLUSTER_VECTOR_SIZE):
    network_df[col] = coords[col]
init_df = network_df.query(f"num_clusters=={INIT_N_CLUSTERS}")
topic_txt = [str(i) for i in init_df["cluster_mode"].unique()]


def tsne_to_cyto(tsne_val, scale_factor=40):
    return int(scale_factor * (float(tsne_val)))


def get_node_list(in_df):
    return [
        {
            "data": {
                "id": str(i),
                "label": str(i),
                "title": row["one_name"],
                "cluster_label": row["cluster_center"],
                "cluster_size": row["cluster_size"],
                "node_size": NODE_SIZE,
            },
            "position": {"x": tsne_to_cyto(row["x"]), "y": tsne_to_cyto(row["y"])},
            "classes": str(row["cluster_id"]),
            "selectable": True,
            "grabbable": False,
        }
        for i, row in in_df.iterrows()
    ]


def get_node_locs(in_df, tsne_perp=DEFAULT_TSNE):
    logger.info(f"Starting dimensionality reduction on {len(in_df)} nodes")

    try:
        node_locs = TSNE(
            n_components=2,
            perplexity=tsne_perp,
            n_iter=300,
            n_iter_without_progress=100,
            learning_rate=150,
            random_state=444,
        ).fit_transform(in_df[list(range(CLUSTER_VECTOR_SIZE))].values)
    except Exception as e:
        logger.error("Something went wrong")
        raise e

    logger.info("Finished dimensionality reduction")

    x_list = node_locs[:, 0]
    y_list = node_locs[:, 1]

    return x_list, y_list


def update_node_data(tsne_perp, in_df):
    (x_list, y_list) = get_node_locs(in_df, tsne_perp=tsne_perp)

    x_range = max(x_list) - min(x_list)
    y_range = max(y_list) - min(y_list)

    scale_factor = int(SCALE_PARAM / (x_range + y_range))
    in_df["x"] = x_list
    in_df["y"] = y_list

    tmp_node_list = get_node_list(in_df)
    for i in range(
        len(in_df)
    ):  # Re-scaling to ensure proper canvas scaling vs node sizes
        tmp_node_list[i]["position"]["x"] = tsne_to_cyto(x_list[i], scale_factor)
        tmp_node_list[i]["position"]["y"] = tsne_to_cyto(y_list[i], scale_factor)

    return tmp_node_list


col_swatch = (
    px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Antique
    + px.colors.qualitative.Prism
    + px.colors.qualitative.Alphabet
)

def_stylesheet = [
    {
        "selector": "." + str(i),
        "style": {"background-color": col_swatch[i], "line-color": col_swatch[i]},
    }
    for i in range(len(network_df["cluster_id"].unique()))
]
def_stylesheet += [
    {
        "selector": "node",
        "style": {"width": "data(node_size)", "height": "data(node_size)"},
    },
]

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Task",
                href="https://medium.com/plotly/exploring-and-investigating-network-relationships-with-plotlys-dash-and-dash-cytoscape-ec625ef63c59?source=friends_link&sk=e70d7561578c54f35681dfba3a132dd5",
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Source Code",
                href="https://github.com/vlade89/agrohack2023",
            )
        ),
    ],
    brand="AgroHack 2023 -  кластеризация сельскохозяйсвтенных профессий",
    brand_href="#",
    color="dark",
    dark=True,
)

topics_html = list()
for topic_html in [
    html.Span([str(i) + ": " + topic_txt[i]], style={"color": col_swatch[i]})
    for i in range(len(topic_txt))
]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

body_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            f"""
                -----
                ##### Данные
                -----
                Для этой кластерии {len(network_df)} резюме было выделено
                {len(network_df.cluster_id.unique())} кластеров с помощью иерархической кластеризации на основе эмбедингов, полученных в результате применения библиотеки 
                [FastText](https://fasttext.cc/).

                Резюме внутри одного кластера имеют один цвет, в разных кластерах - разные. У каждого кластера выделен центральный элемент на основе моды распределения.
                """
                        )
                    ],
                    sm=12,
                    md=6,
                ),
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                -----
                ##### Центры кластеров
                -----
                """
                        ),
                        html.Div(
                            topics_html,
                            id="topics_id",
                            style={
                                "fontSize": 14,
                                "height": "200px",
                                "overflow": "auto",
                            },
                        ),
                    ],
                    sm=12,
                    md=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dcc.Markdown(
                    """
            -----
            ##### Кластеризация
            """
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                cyto.Cytoscape(
                                    id="core_cytoscape",
                                    layout={"name": "preset"},
                                    style={"width": "100%", "height": "600px"},
                                    elements=update_node_data(
                                        DEFAULT_TSNE, in_df=init_df
                                    ),
                                    stylesheet=def_stylesheet,
                                    minZoom=0.06,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Alert(
                                    id="node-data",
                                    children="Выберете вершину чтобы узнать детальную информацию",
                                    color="secondary",
                                )
                            ]
                        ),
                    ],
                    sm=12,
                    md=8,
                ),
                dbc.Col(
                    [
                        dbc.Badge("Кол-во кластеров:", color="info", className="mr-1"),
                        dbc.Container(
                            "Параметр n_clusters по умолчанию: 8 (min: 1, max:100)",
                            id="n_clusters_para",
                            style={"color": "DarkSlateGray", "fontSize": 12},
                        ),
                        dbc.CardGroup(
                            [
                                dcc.Dropdown(
                                    id="n_clusters",
                                    options=[
                                        {"label": k, "value": k}
                                        for k in range(1, 100, 1)
                                    ],
                                    clearable=False,
                                    value=8,
                                )
                            ]
                        ),
                        dbc.Badge(
                            "t-SNE parameters:",
                            color="info",
                            className="mr-1",
                        ),
                        dbc.Container(
                            "Параметр perplexity по умолчанию: 40 (min: 10, max:100)",
                            id="tsne_para",
                            style={"color": "DarkSlateGray", "fontSize": 12},
                        ),
                        dbc.CardGroup(
                            [
                                dcc.Dropdown(
                                    id="tsne_perp",
                                    options=[
                                        {"label": k, "value": k}
                                        for k in range(10, 100, 10)
                                    ],
                                    clearable=False,
                                    value=40,
                                )
                            ]
                        ),
                    ],
                    sm=12,
                    md=4,
                ),
            ]
        ),
    ],
    style={"marginTop": 20},
)

app.layout = html.Div([navbar, body_layout])


@app.callback(
    Output("tsne_para", "children"),
    [Input("tsne_perp", "value")],
)
def update_output(value):
    return f"Current t-SNE perplexity: {value} (min: 10, max:100)"


@app.callback(
    Output("n_clusters_para", "children"),
    [Input("n_clusters", "value")],
)
def update_output(value):
    return f"Current n_clusters: {value} (min: 1, max:100)"


@app.callback(
    dash.dependencies.Output("topics_id", "children"),
    [
        Input("n_clusters", "value"),
    ],
)
def update_topics(n_clusters):
    topics_html = []

    cur_df = network_df.query(f"num_clusters=={n_clusters}")
    topic_txt = [str(i) for i in cur_df["cluster_mode"].unique()]
    for topic_html in [
        html.Span([str(i) + ": " + topic_txt[i]], style={"color": col_swatch[i]})
        for i in range(len(topic_txt))
    ]:
        topics_html.append(topic_html)
        topics_html.append(html.Br())
    return topics_html


@app.callback(
    Output("core_cytoscape", "elements"),
    [
        Input("n_clusters", "value"),
        Input("tsne_perp", "value"),
    ],
)
def filter_nodes(n_clusters, tsne_perp):
    # Generate node list
    cur_df = network_df.query(f"num_clusters=={n_clusters}")
    node_list = update_node_data(tsne_perp, in_df=cur_df)
    return node_list


@app.callback(
    Output("node-data", "children"), [Input("core_cytoscape", "selectedNodeData")]
)
def display_nodedata(datalist):
    contents = "Выберете вершину чтобы узнать детальную информацию"
    if datalist is not None:
        if len(datalist) > 0:
            data = datalist[-1]
            contents = []
            contents.append(html.H5(data["title"].title()))
            contents.append(html.P(f"Центр кластера: {str(data['cluster_label'])}"))
            contents.append(html.P(f"Размер кластера: {data['cluster_size']}"))

    return contents


if __name__ == "__main__":
    app.run_server(debug=False)
