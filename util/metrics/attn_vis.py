import torch

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

def compile_attn(model, loader, device):
    mods = {
        "clinical": "Clinical",
        "path_lang": "Pathology Report",
        "rad_lang": "Radiology Report"
    }
    model.eval()

    for batch in loader:
        for key in batch: batch[key] = batch[key].to(device)
        with torch.inference_mode():
            _, (inp_weights, pair_weights) = model.predict(batch, get_attn = True) #predicted hazards

    inp_weights = {(mods[key[0]], mods[key[1]]): (float(weights[0].mean()), float(weights[1].mean())) for key, weights in inp_weights.items()}
    pair_weights = {key: float(weights.mean()) for key, weights in pair_weights.items()}
    
    return inp_weights, pair_weights

def vis_hat(inp_weights, pair_weights, model_name: str):
    # ---- Colorscale: blue (0) -> purple -> pink/red (1)
    colorscale = [[0.0, "#4A9BE8"], [0.3, "#5A6CD0"], [0.6, "#8E44A8"], [1.0, "#F52A63"]]
    
    box_opacity = 0.8   # 0 = fully transparent, 1 = fully opaque
    
    def color(v, alpha=None):
        """Sample the colorscale and return an rgba() string with the box opacity."""
        rgb = sample_colorscale(colorscale, [v])[0]          # 'rgb(r, g, b)'
        a = box_opacity if alpha is None else alpha
        return rgb.replace("rgb(", "rgba(").replace(")", f", {a})")
    
    # Faded copy of the colorscale so the colorbar matches the boxes
    colorscale_faded = [[p, color(p)] for p in [i / 20 for i in range(21)]]
    
    # ---- Layout parameters
    cell_w, cell_h = 2.0, 0.65
    pair_x = [0.0, 4.3, 8.6]
    bottom_y, mid_y, fusion_y = 1.0, 2.3, 3.6
    top_x = 3.2
    edge = "#3B4A9A"
    font = dict(family="Times New Roman, serif", size=22, color="black")
    
    shapes, annotations = [], []
    
    def cell(x, y, v, w=cell_w, h=cell_h, fill=None, line=edge):
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
                        fillcolor=fill or color(v), line=dict(color=line, width=1.5)))
    
    def line(x0, y0, x1, y1):
        shapes.append(dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                        line=dict(color="black", width=1)))
    
    def arrow(x0, y0, x1, y1):
        annotations.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y",
                                axref="x", ayref="y", text="", showarrow=True,
                                arrowhead=2, arrowsize=1.3, arrowwidth=1,
                                arrowcolor="black"))
    
    # ---- Middle row
    mid_centers = []
    for i, w in enumerate(list(pair_weights.values())):
        x = top_x + i * cell_w
        cell(x, mid_y, w)
        mid_centers.append(x + cell_w / 2)
    
    # ---- Bottom row, labels, connectors
    bracket_y = bottom_y + cell_h + 0.3
    for (names, vals), x, cx_mid in zip(list(inp_weights.items()), pair_x, mid_centers):
        centers = []
        for j, (name, v) in enumerate(zip(names, vals)):
            cx = x + j * cell_w
            cell(cx, bottom_y, v)
            c = cx + cell_w / 2
            centers.append(c)
            annotations.append(dict(x=c, y=bottom_y - 0.6, text=name,
                                    showarrow=False, font=font))
            line(c, bottom_y + cell_h, c, bracket_y)
        line(min(centers + [cx_mid]), bracket_y, max(centers + [cx_mid]), bracket_y)
        arrow(cx_mid, bracket_y, cx_mid, mid_y)
    
    # ---- Fusion box
    fw, fh = 2.0, 0.65
    fx = top_x + 1.5 * cell_w - fw / 2
    cell(fx, fusion_y, None, w=fw, h=fh, fill="#E6E6E6", line="#6B6B3A")
    annotations.append(dict(x=fx + fw / 2, y=fusion_y + fh / 2,
                            text="<i>x</i><sup><i>fusion</i></sup>",
                            showarrow=False, font=font))
    arrow(fx + fw / 2, mid_y + cell_h, fx + fw / 2, fusion_y)
    
    # ---- Invisible trace that only exists to draw the colorbar
    colorbar_trace = go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale=colorscale_faded, cmin=0, cmax=1, color=[0],
                    showscale=True,
                    colorbar=dict(tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                thickness=18, len=1.0, outlinewidth=1,
                                tickfont=dict(family="Times New Roman, serif", size=16))),
        hoverinfo="none", showlegend=False,
    )
    
    fig = go.Figure(colorbar_trace)
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        width=1300, height=480, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.2, 14.0], visible=False),
        yaxis=dict(range=[0, 4.5], visible=False, scaleanchor="x", scaleratio=1),
    )
    
    fig.write_image(f"vises/attn/{model_name} hat_attn_weights.pdf")