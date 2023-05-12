import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash import dash_table
import plotly.express as px
from dash import dcc
from plotly.subplots import make_subplots
import numpy as np
from dash import html
from dash.dependencies import Output
from dash.dependencies import Input, Output


df = pd.read_csv("dataset/Train_Test_Size.csv")
# Filter the data to include only Train and Test
train_test_df = df[df["label"].isin(["Training_Size", "Testing_Size"])]
raw_df = pd.read_csv("dataset/TinTucGocVNIndex.csv")
clean_df = pd.read_csv("dataset/NewsAll_Flag.csv")
dfAccuracy = pd.read_csv("dataset/KQTNPT.csv")
dfLossAccuracy = pd.read_csv("dataset/Loss_Accuracy.csv")
dfLoss3 = pd.read_csv("dataset/Loss02.csv")
df0405 = pd.read_csv("dataset/Prediction0405.csv")
df0505 = pd.read_csv("dataset/Prediction0505.csv")
df0805 = pd.read_csv("dataset/Prediction0805.csv")
df0905 = pd.read_csv("dataset/Prediction0905.csv")
flag_df = pd.read_csv("dataset/NewsAndFlag.csv")
dfTrainTestFlag = pd.read_csv("dataset/Train_Test_Flag.csv")
dfTime = pd.read_csv("dataset/TimeTN.csv")

# Calculate the grand total
grand_total = train_test_df["Value"].sum()

# Lấy danh sách tên model
model_names = dfAccuracy["Model_name"].unique()

# Thêm option "All Models" vào danh sách tên model
model_names = np.append(model_names, "All Models")

# Tạo dropdown options cho model selection
dropdown_options = [{"label": name, "value": name} for name in model_names]
# Create figure
fig = px.bar(
    dfLossAccuracy,
    x="Model_name",
    y=["test_loss", "test_accuracy"],
    barmode="group",
    title="Loss and Accuracy by Model (Test dataset)",
)

fig1 = px.bar(
    dfTrainTestFlag,
    x="label",
    y=["Pos", "Neg"],
    barmode="group",
    title="Pos_Neg on Train_Test Dataset",
)

fig2 = px.bar(
    dfTime,
    x="Model_name",
    y="elapsed_time",
    title="Training time (minutes) for each model",
)

# Define the pie chart traces
traces = [
    go.Pie(labels=train_test_df["label"], values=train_test_df["Value"]),
    go.Pie(labels=["Total"], values=[grand_total])
]

traces1 = [
    go.Pie(labels=flag_df['Lable'],values=flag_df['Values'], textinfo='percent')
]
available_models = dfLoss3["Model_name"].unique()


# Define function to create pie chart
def create_pie_chart(df, model):
    colors = ["#ff3737", "#0f0"]
    values = [sum(df["value"] < 0.5), sum(df["value"] >= 0.5)]
    labels = ["<0.5", ">=0.5"]
    pie_chart = go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hoverinfo="label+percent",
        textinfo="value",
        textfont=dict(size=18),
        hole=0.5,
    )
    layout = go.Layout(
        title=model,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", x=0.7, y=0.2),
    )
    fig = go.Figure(data=[pie_chart], layout=layout)
    return fig


external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?" "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]


def create_pie_chart(df, model):
    colors = ["#ff3737", "#0f0"]
    values = [sum(df["value"] < 0.5), sum(df["value"] >= 0.5)]
    labels = ["Giảm", "Tăng"]
    pie_chart = go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hoverinfo="label+percent",
        textinfo="value",
        textfont=dict(size=18),
        hole=0.5,
    )
    layout = go.Layout(
        title=model,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", x=0.7, y=0.2),
    )
    fig = go.Figure(data=[pie_chart], layout=layout)
    return fig


app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "VNINDEX Movements Prediction"


app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="📈", className="header-emoji"),
                html.H1(children="VNINDEX Movements Prediction", className="header-title"),
                html.P(
                    children=("Đỗ Chí Bảo - " "Nguyễn Thị Mỹ Hải - " "Hoàng Minh Khiêm"),
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Dataset",
                            children=[
                                html.H2("Raw data - 58.238"),
                                dash_table.DataTable(
                                    id="raw-and-clean-table",
                                    columns=[
                                        {"name": i, "id": i} for i in raw_df.columns
                                    ],
                                    data=raw_df.head(10).to_dict("records"),
                                    style_cell={
                                        "textAlign": "center",
                                        "width": "auto",
                                        "minWidth": "50px",
                                        "maxWidth": "200px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                    },
                                    page_size=10,
                                    fixed_rows={"headers": True, "data": 0},
                                    style_table={
                                        "maxHeight": "500px",
                                        "overflowY": "scroll",
                                        "width": "100%",
                                    },
                                ),
                                html.H2("Clean data - 49.780"),
                                dash_table.DataTable(
                                    id="clean-table",
                                    columns=[
                                        {"name": i, "id": i} for i in clean_df.columns
                                    ],
                                    data=clean_df.head(10).to_dict("records"),
                                    style_cell={
                                        "textAlign": "center",
                                        "width": "auto",
                                        "minWidth": "50px",
                                        "maxWidth": "200px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                    },
                                    page_size=10,
                                    fixed_rows={"headers": True, "data": 0},
                                    style_table={
                                        "maxHeight": "500px",
                                        "overflowY": "scroll",
                                        "width": "100%",
                                    },
                                ),
                                dcc.Graph(
                                    id="pie-chart",
                                    figure={
                                        "data": traces,
                                        "layout": go.Layout(
                                            title="Train_Test Size Pie Chart"
                                        ),
                                    },
                                ),
                                 dcc.Graph(
                                    id="pie-chart6",
                                    figure={
                                        "data": traces1,
                                        "layout": go.Layout(
                                            title="The ratio of Pos_Neg in the dataset"
                                        ),
                                    },
                                ),
                                html.H1("The ratio of Pos_Neg on Train_Test Dataset"),                             
                                dcc.Graph(figure=fig1),
                                ],
                        ),
                        dcc.Tab(
                            label="Training process",
                            children=[
                                html.H1("Loss and Accuracy by Model (Test dataset)"),
                                dcc.Graph(figure=fig),
                                dcc.Graph(figure=fig2),
                                html.Label("Model Selection"),
                                dcc.Dropdown(
                                    id="model-selection",
                                    options=dropdown_options,
                                    value="BiGRU",
                                    clearable=False,
                                ),
                                html.H1(children="Accuracy Comparison"),
                                dcc.Graph(id="accuracy-graph", figure={}),
                                html.H1("Loss Chart"),
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    options=[
                                        {"label": model, "value": model}
                                        for model in ["All Models"]
                                        + list(available_models)
                                    ],
                                    value="All Models",
                                ),
                                html.Div(id="charts-container"),
                            ],
                        ),
                        dcc.Tab(
                            label="Prediction",
                            children=[
                                html.H1("Stock Prediction"),
                                dcc.Dropdown(
                                    id="date-dropdown",
                                    options=[
                                        {"label": "04/05", "value": "0405"},
                                        {"label": "05/05", "value": "0505"},
                                        {"label": "08/05", "value": "0805"},
                                        {"label": "09/05", "value": "0905"},
                                    ],
                                    value="0405",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="pie-chart-1", className="six columns"
                                        ),
                                        dcc.Graph(
                                            id="pie-chart-2", className="six columns"
                                        ),
                                    ],
                                    className="row",
                                    style=dict(display="flex", width="100%"),
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="pie-chart-3", className="six columns"
                                        ),
                                        dcc.Graph(
                                            id="pie-chart-4", className="six columns"
                                        ),
                                    ],
                                    className="row",
                                    style=dict(display="flex", width="100%"),
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="pie-chart-5", className="six columns"
                                        ),
                                    ],
                                    className="row",
                                    style=dict(display="flex", width="100%")
                                    )                           
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

# Callback function to update the graph based on the selected model


@app.callback(Output("accuracy-graph", "figure"), Input("model-selection", "value"))
def update_graph(model):
    if model == "All Models":
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, subplot_titles=model_names[:5]
        )
        for i, name in enumerate(model_names[:5]):
            df = dfAccuracy[dfAccuracy["Model_name"] == name]
            fig.add_trace(
                go.Scatter(
                    x=df["Epoch"], y=df["train_accuracy"], mode="lines", name="accuracy"
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(height=1200, showlegend=False)

        fig.update_yaxes(range=[0, 1])
    else:
        # Hiển thị biểu đồ cho mô hình được lựa chọn
        df = dfAccuracy[dfAccuracy["Model_name"] == model]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Epoch"], y=df["train_accuracy"], mode="lines", name="train_accuracy"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Epoch"],
                y=df["val_accuracy"],
                mode="lines",
                name="val_accuracy",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Epoch"],
                y=df["test_accuracy"],
                mode="lines",
                name="test_accuracy",
            )
        )
        fig.update_layout(
            title=model + " Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)"
        )
        fig.update_layout(height=500, showlegend=True)
    return fig


@app.callback(
    Output("charts-container", "children"), [Input("model-dropdown", "value")]
)
def update_loss_graph(model):
    # Nếu chọn tất cả các model
    if model == "All Models":
        # Tạo danh sách các biểu đồ
        charts = []
        for model_name in available_models:
            # Lấy dữ liệu cho model này
            model_data = dfLoss3[dfLoss3["Model_name"] == model_name]
            # Tạo biểu đồ
            fig = px.line(
                model_data,
                x="Epoch",
                y=["val_loss", "train_loss"],
                title=model_name,
                labels={"value": "loss"},
            )
            fig.update_yaxes(range=[0.5, 1])
            # Thêm biểu đồ vào danh sách
            charts.append(dcc.Graph(figure=fig))

        # Đặt kích thước của mỗi biểu đồ
        chart_style = {
            "width": "30%",
            "height": "300px",
            "display": "inline-block",
            "padding": "10px",
        }
        chart_rows = []
        for i in range(0, len(charts) - 2, 2):
            # Tạo một hàng mới
            row = html.Div(
                [charts[i], charts[i + 1]], style={"display": "flex"}
            )
            chart_rows.append(row)

        # Thêm hai biểu đồ cuối cùng vào hàng cuối cùng
        if len(charts) % 2 == 1:
            row = html.Div([charts[-1]], style={"display": "flex"})
            chart_rows.append(row)

        # Đặt tất cả các hàng vào một lưới dữ liệu
        return html.Div(chart_rows)
    else:
        # Lấy dữ liệu cho model được chọn
        model_data = dfLoss3[dfLoss3["Model_name"] == model]
        # Tạo biểu đồ
        fig = px.line(
            model_data,
            x="Epoch",
            y=["val_loss", "train_loss"],
            title=model,
            labels={"value": "loss"},
        )
        fig.update_yaxes(range=[0.5, 1])
        return dcc.Graph(figure=fig)


@app.callback(
    Output("pie-chart-1", "figure"),
    Output("pie-chart-2", "figure"),
    Output("pie-chart-3", "figure"),
    Output("pie-chart-4", "figure"),
    Output("pie-chart-5", "figure"),
    Input("date-dropdown", "value"),
)
def update_pie_charts(date):
    if date == "0405":
        df = df0405
    elif date == "0505":
        df = df0505
    elif date == "0805":
        df = df0805
    else:
        df = df0905
    pie_chart_1 = create_pie_chart(df[df["Model_name"] == "RNN"], "RNN")
    pie_chart_2 = create_pie_chart(df[df["Model_name"] == "GRU"], "GRU")
    pie_chart_3 = create_pie_chart(df[df["Model_name"] == "LSTM"], "LSTM")
    pie_chart_4 = create_pie_chart(df[df["Model_name"] == "BGRU"], "BGRU")
    pie_chart_5 = create_pie_chart(df[df["Model_name"] == "BiLSTM"], "BiLSTM")
    return pie_chart_1, pie_chart_2, pie_chart_3, pie_chart_4, pie_chart_5


if __name__ == "__main__":
    app.run_server(debug=True)
