import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

def extrair_ano(cat_number):
    return int('20' + cat_number.split('-')[1][:2])

def criar_df_area_empilhada(df,taxon,ano='year'):
    df_area = df.copy()
    df_area = df_area.groupby([taxon,ano]).size().reset_index(name='counts')
    df_area = df_area.sort_values(by=ano)
    df_area[ano] = pd.to_numeric(df_area[ano])
    df_area = df_area.pivot_table(index=taxon, columns=ano, values='counts', fill_value=0)
    df_area = df_area.cumsum(axis=1)
    df_area = df_area.reset_index()

    df_area_long = pd.melt(df_area, id_vars=[taxon], var_name=ano, value_name='value')
    last_year_order_counts = df_area_long[df_area_long[ano] == df_area_long[ano].max()].groupby(taxon).size()
    df_area_long_sorted = df_area_long.set_index(taxon).loc[last_year_order_counts.index].reset_index()

    sorted_order = df_area_long_sorted[df_area_long_sorted[ano] == max(df_area_long_sorted[ano])].groupby(taxon)['value'].sum().sort_values(ascending=False).index

    return df_area_long_sorted,sorted_order

def criar_df_linha(df,taxon,ano='year'):
    df_linha = df.copy()
    df_linha = df_linha.groupby([taxon, ano]).size().reset_index(name='counts')
    df_linha = df_linha.sort_values(by=ano)
    df_linha[ano] = pd.to_numeric(df_linha[ano])
    df_linha = df_linha.pivot_table(index=taxon, columns=ano, values='counts', fill_value=0)
    df_linha_soma = df_linha.cumsum(axis=1)
    df_linha_soma = df_linha_soma.reset_index()
    df_linha = df_linha.reset_index()

    sorted_order = list_sorted_order(df_linha_soma,taxon,ano)

    df_linha[taxon] = pd.Categorical(df_linha[taxon], categories=sorted_order, ordered=True)
    df_linha = df_linha.sort_values(by=taxon)
    return df_linha,sorted_order

def list_sorted_order(df,taxon,ano):
    df_area_long = pd.melt(df, id_vars=[taxon], var_name=ano, value_name='value')
    last_year_order_counts = df_area_long[df_area_long[ano] == df_area_long[ano].max()].groupby(taxon).size()
    df_area_long_sorted = df_area_long.set_index(taxon).loc[last_year_order_counts.index].reset_index()

    sorted_order = df_area_long_sorted[df_area_long_sorted[ano] == max(df_area_long_sorted[ano])].groupby(taxon)['value'].sum().sort_values(ascending=False).index
    return sorted_order

def calcula_condicional_ausentes_porcentagem(df, previous_level, current_level):
    conditioned_df = df[df[previous_level].notnull()]
    return (conditioned_df[current_level].isnull().sum() / len(conditioned_df)) * 100

def create_brazil_bar_chart(df):
    # df_agrupado = df.groupby(['stateProvince', 'ano']).size().reset_index(name='counts')

    # Ordene os dados pelo ano
    df_agrupado = df.sort_values(by='counts')

    # Crie o gráfico de barras empilhadas
    fig_brasil = px.bar(
        df_agrupado,
        color='stateProvince',
        y='counts',
        x='ano',
        title='Contagem de Coletas por Estado no Brasil (Empilhado por Ano)',
        labels={'counts': 'Contagem de Coletas', 'stateProvince': 'Estado', 'ano': 'Ano'},
        barmode='stack'
    )
    return fig_brasil
url = 'https://raw.githubusercontent.com/mariohenriique/visualizacaodados/main/planilha_unificada_alterada.csv'
df = pd.read_csv(url,low_memory=False)

df['specie'] = df.apply(lambda row: row['genus'] + ' ' + row['specificEpithet'] if not pd.isna(row['genus']) and not pd.isna(row['specificEpithet']) else None, axis=1)

df['year'] = df['eventDate'].str.split('/').str[0].str.split('-').str[0]
df['mounth'] = df['eventDate'].str.split('/').str[0].str.split('-').str[1]
df['day'] = df['eventDate'].str.split('/').str[0].str.split('-').str[2]
df['ano'] = df['catalogNumber'].apply(lambda x: extrair_ano(x) if isinstance(x, str) else None)

# Filtrando as colunas sem informações
col = ['continent', 'country','stateProvince','county','locality']
columns_to_fill = ['kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'genus', 'specie']
df_sem_classificao = df.assign(**{col: df[col].fillna('sem classificação') for col in columns_to_fill})
df_treemap_classificacao = df.assign(**{col: df[col].fillna('sem classificação') for col in columns_to_fill})

df_paises = df.assign(**{col: df[col].fillna('sem informação do local') for col in col})
df_sem_brasil = df_paises[df_paises['country'] != 'Brasil']
df_brasil = df_paises[df_paises['country'] == 'Brasil']
df_sem_mg = df_brasil[df_brasil['stateProvince'] != 'Minas Gerais']

treemap_class_taxon = px.treemap(df_treemap_classificacao, path=['kingdom','phylum', 'class','order','superfamily','family','subfamily','genus','specie'],color='order')
treemap_class_taxon.update_layout(margin = dict(t=50, l=25, r=25, b=25))

treemap_paises = px.treemap(df_paises, path=[px.Constant("world"), 'continent', 'country','stateProvince','county','locality'])
treemap_paises.update_layout(margin = dict(t=50, l=25, r=25, b=25))

tax_levels = ['order', 'superfamily', 'family', 'subfamily', 'genus','subgenus','specie']
area_empilhada_dict = {}
sorted_orders = {}

for tax in tax_levels:
    # DataFrame com a coluna 'ano'
    df_tax_ano, sorted_tax_ano = criar_df_area_empilhada(df_sem_classificao, tax, 'ano')
    area_empilhada_ano = px.area(df_tax_ano, x='ano', y='value', color=tax,
                      title=f'Contagem de Coletas por {tax.capitalize()} (Empilhado por Ano)',
                      labels={'value': 'Contagem de Coletas', tax: tax.capitalize(), 'ano': 'Ano'},
                      category_orders={tax: sorted_tax_ano})

    # DataFrame com a coluna 'year'
    df_tax_year, sorted_tax_year = criar_df_area_empilhada(df_sem_classificao, tax, 'year')
    area_empilhada_year = px.area(df_tax_year, x='year', y='value', color=tax,
                       title=f'Contagem de Coletas por {tax.capitalize()} (Empilhado por Ano)',
                       labels={'value': 'Contagem de Coletas', tax: tax.capitalize(), 'year': 'Ano'},
                       category_orders={tax: sorted_tax_year})

    area_empilhada_dict[tax] = {'ano': area_empilhada_ano, 'year': area_empilhada_year}
    sorted_orders[tax] = sorted_tax_ano

# Criar uma figura de plotly.graph_objects
grafico_area_empilhada = go.Figure()

# categories = []
for tax in tax_levels:
    for trace in area_empilhada_dict[tax]['ano'].data:
        if trace.name == 'sem classificação':
            trace.visible = 'legendonly'
        grafico_area_empilhada.add_trace(trace)
        grafico_area_empilhada.data[-1].visible = False  # Inicialmente definir visível como False

    for trace in area_empilhada_dict[tax]['year'].data:
        grafico_area_empilhada.add_trace(trace)
        grafico_area_empilhada.data[-1].visible = False  # Inicialmente definir visível como False
        # categories.append((tax, 'year'))

# Definir a visibilidade inicial (genus e ano)
for i in range(len(area_empilhada_dict['order']['ano'].data)):
    grafico_area_empilhada.data[i].visible = True

histograma_tax_dict = {}
for tax in tax_levels:
    df_tax_filtered = df_sem_classificao[df_sem_classificao[tax] != 'sem classificação']
    # df_tax_filtered[tax] = df_tax_filtered[tax].apply(lambda x: [s.strip() for s in x.split(' | ')] if isinstance(x, str) else x)

    # # Explodir a lista resultante em várias linhas e criar uma nova coluna
    # df_tax_filtered = df_tax_filtered.explode(tax).reset_index(drop=True)
    # DataFrame com a contagem dos itens classificados
    df_tax_count = df_tax_filtered[tax].value_counts()
    hist_tax = px.bar(df_tax_count,log_y=True)
    hist_tax.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df_tax_count.index,  # Define os valores dos rótulos
            ticktext = [str(label)[:15] for label in df_tax_count.index]  # Limita o comprimento dos rótulos
        )
    )
    histograma_tax_dict[tax] = hist_tax

# Fazer um de linha para as coletas (só tem pra identificacao)
linha_tax_dict = {}

for tax in tax_levels:
    df_linha, sorted_orders = criar_df_linha(df_sem_classificao, tax)

    todos_anos = range(max(1984, df_linha.iloc[:, 1:].values.min()), df_linha.columns[1:].max() + 1)
    anos_faltantes = [ano for ano in todos_anos if ano not in df_linha.columns[1:]]
    
    traces=[]
    linha_tax_unico_dict = {}  # Dicionário interno para armazenar os gráficos de linha por categoria
    for index, row in df_linha.iterrows():
        visibility = 'legendonly' if row[tax] == 'sem classificação' else True
        trace = go.Scatter(x=df_linha.columns[1:], y=row.values[1:], mode='lines+markers', name=row[tax], visible=visibility)
        traces.append(trace)
        grafico_linha_unico = px.line(row,markers=True)
        linha_tax_unico_dict[row[tax]] = grafico_linha_unico

    layout = go.Layout(
        xaxis=dict(title='Ano'),
        yaxis=dict(
            title='Contagem de Coletas',
            range=[0 - (df_linha.iloc[:, 1:].values.max() * 0.03), df_linha.iloc[:, 1:].values.max() * 1.03]
        )
    )

    grafico_linha_todos = go.Figure(data=traces, layout=layout)
    for index, valor in enumerate(anos_faltantes):
        grafico_linha_todos.add_shape(
            type="rect",
            x0=valor,
            y0=0,
            x1=valor,
            y1=df_linha.iloc[:, 1:].values.max() * 100,  # Altura máxima com base nos dados
            opacity=0.3,
            line=dict(color="Gray", width=10)
        )

    grafico_linha_todos.update_layout(xaxis=dict(showgrid=False))
    linha_tax_unico_dict['all'] = grafico_linha_todos
    linha_tax_dict[tax] = linha_tax_unico_dict

df_percent_class = df[['order','Coorte ou Supercoorte','superfamily','family','subfamily','genus','subgenus','specie']]

valores_null_porcentagem = (df_percent_class.isnull().sum() / len(df_percent_class)) * 100

grafico_valores_null = px.bar(valores_null_porcentagem)

# Lista de níveis hierárquicos na ordem desejada
taxon_levels = ['class','order', 'Coorte ou Supercoorte','superfamily', 'family','genus', 'specie']

# Dicionário para armazenar as porcentagens
porcentagem_valores_ausentes = {}

# Calcular as porcentagens para cada nível hierárquico
for i in range(1, len(taxon_levels)):
    nivel_anterior = taxon_levels[i - 1]
    nivel_atual = taxon_levels[i]
    porcentagem_valores_ausentes[nivel_atual] = calcula_condicional_ausentes_porcentagem(df, nivel_anterior, nivel_atual)

# Criar um DataFrame a partir do dicionário de porcentagens
df_porcentagem_valores_ausentes = pd.DataFrame(list(porcentagem_valores_ausentes.items()), columns=['nivel', 'valor'])

grafico_valores_null_dependente = px.bar(df_porcentagem_valores_ausentes,x='nivel',y='valor')

df_tax_unicos = df[tax_levels].nunique()

grafico_unicos_taxon = px.bar(df_tax_unicos)

barras_coletores_dict = {}
df_coletores = df.copy()
df_coletores = df_coletores[['recordedBy','year','ano']]

df_coletores['recordedBy'] = df_coletores['recordedBy'].dropna().apply(lambda x: [s.strip() for s in x.split(' | ')] if isinstance(x, str) else x)

# Explodir a lista resultante em várias linhas e criar uma nova coluna
df_coletores = df_coletores.explode('recordedBy').reset_index(drop=True)

df_coletores_grouped_coleta = df_coletores.groupby('year')['recordedBy'].nunique().reset_index()
df_coletores_grouped_identificacao = df_coletores.groupby('ano')['recordedBy'].nunique().reset_index()

# Criar o gráfico de barras
grafico_barras_coletores_coleta = px.bar(df_coletores_grouped_coleta,x='year',y='recordedBy')
barras_coletores_dict['year'] = grafico_barras_coletores_coleta

grafico_barras_coletores_identificacao = px.bar(df_coletores_grouped_identificacao,x='ano',y='recordedBy')
barras_coletores_dict['ano'] = grafico_barras_coletores_identificacao

# Verificar os dados de cada um dos coletores
barras_equipe_dict = {}
df_filtrado_equipe = df_coletores[df_coletores['recordedBy'].str.contains('Equipe', na=False)]

# Contar a quantidade de ocorrências em cada ano
df_contagem_coleta = df_filtrado_equipe.groupby(['year', 'recordedBy']).size().reset_index(name='count')
df_contagem_identificacao = df_filtrado_equipe.groupby(['ano', 'recordedBy']).size().reset_index(name='count')

grafico_barras_equipe_coleta = px.bar(df_contagem_coleta, x='year', y='count', color='recordedBy', barmode='stack',
             labels={'year': 'Ano', 'count': 'Contagem', 'recordedBy': 'Registrado por'},
             title='Contagem de ocorrências por ano e registrado por')
barras_equipe_dict['year'] = grafico_barras_equipe_coleta

grafico_barras_equipe_identificacao = px.bar(df_contagem_identificacao, x='ano', y='count', color='recordedBy', barmode='stack',
             labels={'year': 'Ano', 'count': 'Contagem', 'recordedBy': 'Registrado por'},
             title='Contagem de ocorrências por ano e registrado por')
barras_equipe_dict['ano'] = grafico_barras_equipe_identificacao

barras_paises_dict = {}
df_ano = df.copy()

df_agrupado_cole = df_ano.groupby(['country', 'year']).size().reset_index(name='counts')
df_agrupado_iden = df_ano.groupby(['country', 'ano']).size().reset_index(name='counts')

# Ordene os dados pelo ano
df_agrupado_cole = df_agrupado_cole.sort_values(by='counts')
df_agrupado_iden = df_agrupado_iden.sort_values(by='ano')

# Crie o gráfico de barras empilhadas
grafico_barras_paises = px.bar(df_agrupado_cole, color='country', y='counts', x='year', title='Contagem de Coletas por País (Empilhado por Ano)',
             labels={'counts': 'Contagem de Coletas', 'country': 'País', 'year': 'Ano'},
             barmode='stack')
grafico_barras_paises_ano = px.bar(df_agrupado_iden, color='country', y='counts', x='ano', title='Contagem de Coletas por País (Empilhado por Ano)',
             labels={'counts': 'Contagem de Coletas', 'country': 'País', 'year': 'Ano'},
             barmode='stack')

barras_paises_dict['year'] = grafico_barras_paises
barras_paises_dict['ano'] = grafico_barras_paises_ano

df_state = df.copy()
df_state = df_state[df_state['country']=='Brasil']
df_state = df_state.groupby(['stateProvince', 'ano']).size().reset_index(name='counts')

# Inicializar o aplicativo JupyterDash
app = Dash(__name__)
server = app.server
# Layout do aplicativo
app.layout = html.Div([
    dcc.Graph(id='treemap_brasil', figure=treemap_class_taxon),
    dcc.Dropdown(
        id='tax-selector',
        options=[{'label': tax.capitalize(), 'value': tax} for tax in tax_levels],
        value='order'
    ),
    dcc.Dropdown(
        id='time-selector',
        options=[{'label': 'Identificação', 'value': 'ano'}, {'label': 'Coleta', 'value': 'year'}],
        value='ano',
    ),
    dcc.Graph(id='graph'),
    dcc.Graph(id='graph1'),
    dcc.Dropdown(id='line-tax-dropdown'),
    dcc.Graph(id='line-graph'),
    dcc.Graph(id='unique_tax', figure=grafico_unicos_taxon),
    dcc.Graph(id='percent_null', figure=grafico_valores_null),
    dcc.Graph(id='percent_null_denpendent', figure=grafico_valores_null_dependente),
    dcc.Graph(id='coletores'),
    dcc.Graph(id='equipe'),

    dcc.Graph(id='treemap_paises',figure=treemap_paises),
    html.Label("Selecione o País:"),
    dcc.Dropdown(
        id='country-filter',
        options=[{'label': c, 'value': c} for c in df_agrupado_cole['country'].unique()],
        value=df_agrupado_cole['country'].unique().tolist(),  # Valor padrão: todos os países
        multi=True  # Permite seleção múltipla
    ),    
    dcc.Graph(id='barras-country',clear_on_unhover=True),

    html.Div(id='grafico-brasil-container', children=[
        dcc.Dropdown(
            id='state-filter',
            options=[{'label': c, 'value': c} for c in df_state['stateProvince'].unique()],
            value=df_state['stateProvince'].unique().tolist(),  # Valor padrão: todos os países
            multi=True  # Permite seleção múltipla
        ),
        dcc.Graph(id='grafico-brasil',clear_on_unhover=True)
    ]),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,  # Número de linhas por página
        filter_action='native',
        style_table={'overflowX': 'auto'},  # Permite rolagem horizontal
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'whiteSpace': 'normal',
            'height': 'auto'
        }
    ),
])

line_dropdown_value = 'all'

@app.callback(
    [Output('line-tax-dropdown', 'options'),
     Output('line-tax-dropdown', 'value')],
    [Input('tax-selector', 'value')]
)
def update_line_tax_dropdown(selected_tax):
    global line_dropdown_value
    # Obter as opções de itens disponíveis no gráfico de linha para o nível taxonômico selecionado
    line_tax_dropdown_options = [{'label': f'Mostrar todas as {selected_tax}', 'value': nome} if nome =='all' else {'label': nome, 'value': nome} for nome in linha_tax_dict[selected_tax]]

    # Definir o valor inicial do dropdown para a seleção armazenada
    initial_value = 'all'
    
    return line_tax_dropdown_options, initial_value

@app.callback(
    [Output('graph', 'figure'),
     Output('graph1', 'figure'),
     Output('line-graph', 'figure'),
     Output('coletores', 'figure'),
     Output('equipe', 'figure'),
     Output('barras-country', 'figure'),
     Output('grafico-brasil', 'figure'),
     Output('grafico-brasil-container', 'style')],
    [Input('tax-selector', 'value'),
     Input('time-selector', 'value'),
     Input('line-tax-dropdown', 'value'),
     Input('barras-country', 'hoverData'),
     Input('grafico-brasil', 'hoverData'),
     Input('country-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_figure(tax_selection, time_selection, line_tax_selection, hoverData, brasil_hoverData, selected_countries, selected_state):
    global current_tax_selection, current_time_selection, line_dropdown_value
    current_tax_selection = tax_selection
    current_time_selection = time_selection

    # Atualize o gráfico de linha apenas se a seleção mudou
    if line_tax_selection != line_dropdown_value:
        line_dropdown_value = line_tax_selection

    # Access the appropriate area_empilhada_dict entry and retrieve the figure
    # output_text = f'Selecionado: {line_tax_selection} - {clickData} {hoverData}'
    figure_to_return = area_empilhada_dict[tax_selection][time_selection]
    histograma_return = histograma_tax_dict[tax_selection]
    line_figure = linha_tax_dict[tax_selection][line_tax_selection] if line_tax_selection in linha_tax_dict[tax_selection] else linha_tax_dict[tax_selection]
    barras_coletores_return = barras_coletores_dict[time_selection]
    barras_equipe_return = barras_equipe_dict[time_selection]

    # Filtrar o gráfico de barras para os países selecionados
    if time_selection == 'year':
        df_agrupado_cole = df.groupby(['country', 'year']).size().reset_index(name='counts')
        if selected_countries:
            df_agrupado_cole = df_agrupado_cole[df_agrupado_cole['country'].isin(selected_countries)]
        fig = px.bar(df_agrupado_cole, color='country', y='counts', x='year',
                     title='Contagem de Coletas por País (Empilhado por Ano)',
                     labels={'counts': 'Contagem de Coletas', 'country': 'País', 'year': 'Ano'},
                     barmode='stack')
    else:
        df_agrupado_iden = df.groupby(['country', 'ano']).size().reset_index(name='counts')
        if selected_countries:
            df_agrupado_iden = df_agrupado_iden[df_agrupado_iden['country'].isin(selected_countries)]
        fig = px.bar(df_agrupado_iden, color='country', y='counts', x='ano',
                     title='Contagem de Coletas por País (Empilhado por Ano)',
                     labels={'counts': 'Contagem de Coletas', 'country': 'País', 'ano': 'Ano'},
                     barmode='stack')

    if hoverData and 'points' in hoverData:
        hovered_country = hoverData['points'][0]['curveNumber']
        for trace in fig.data:
            trace['marker']['opacity'] = 0.5
        fig.data[hovered_country]['marker']['opacity'] = 1.0
    else:
        for trace in fig.data:
            trace['marker']['opacity'] = 1.0

    if 'Brasil' in selected_countries:
        df_filtered_brasil = df_state[df_state['stateProvince'].isin(selected_state)]
        grafico_brasil = create_brazil_bar_chart(df_filtered_brasil)
        brasil_style = {'display': 'block'}
        if brasil_hoverData and 'points' in brasil_hoverData:
            hovered_index = brasil_hoverData['points'][0]['curveNumber']
            for i, trace in enumerate(grafico_brasil.data):
                if i == hovered_index:
                    trace.marker.opacity = 1.0
                else:
                    trace.marker.opacity = 0.5
        else:
            for trace in grafico_brasil.data:
                trace.marker.opacity = 1.0
    else:
        grafico_brasil = go.Figure()  # Gráfico vazio quando o Brasil não está selecionado
        brasil_style = {'display': 'none'}

    return figure_to_return, histograma_return, line_figure, barras_coletores_return, barras_equipe_return, fig, grafico_brasil, brasil_style

if __name__ == '__main__':
    app.run_server(debug=True)
