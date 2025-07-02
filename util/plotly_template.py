import plotly.express as px
import plotly.graph_objects as go


def setup_plotly_theme(name: str = 'seaborn'):
    import plotly.io as pio

    go.layout.Template()
    pio.templates['custom'] = go.layout.Template(
        # Fix the graph size
        layout={'width': 800, 'height': 600, 'autosize': False},
        # Watermark
        layout_annotations=[dict(
            name="NameWatermark",
            text="Listenzcc",
            textangle=-30,
            opacity=0.05,
            font=dict(color="#110000", size=100),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        ])
    pio.templates.default = '+'.join([name, 'custom'])
    return pio.templates[pio.templates.default]
