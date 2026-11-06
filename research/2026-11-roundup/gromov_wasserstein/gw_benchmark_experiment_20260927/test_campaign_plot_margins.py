"""Visible gap intervals keep padding after switching to signed log scale."""
import numpy as np
import matplotlib.pyplot as plt
import summarize_campaign as module


def test_gap_error_bars_have_margin_in_display_coordinates(monkeypatch, tmp_path):
    def stats(median, low=None, high=None):
        return dict(count=5, median=median, q25=median if low is None else low,
                    q75=median if high is None else high)
    curves=[]
    for method,gap,low,high in [('fw',0.,0.,0.),('native-sampled-gw',.003,.0029,.0031)]:
        curves.append(dict(phase='final_product',initialization='product',family='synthetic',n=512,
                           noise=0.,method=method,problem_count=5,points=[dict(observed_units=5,
                           coverage_complete=True,metrics={'time_to_output':stats(1.),
                           'actual_solver_seconds':stats(1.),'rounded_loss':stats(.04),
                           'original_fw_gap':stats(gap,low,high),'col_l1':stats(0.),'col_l2':stats(0.)})]))
    checked=[]
    def inspect(fig,out,name,files,title):
        try:
            if name.endswith('_original_gap'):
                ax=fig.axes[0];fig.canvas.draw()
                ys=ax.transData.transform(np.array([[1.,0.],[1.,.0031]]))[:,1]
                fraction=(ys-ax.bbox.y0)/ax.bbox.height
                assert ax.get_yscale()=='symlog'
                assert .02 < fraction[0] < .1
                assert .9 < fraction[1] < .98
                checked.append(name)
        finally:
            plt.close(fig)
    monkeypatch.setattr(module,'save_figure',inspect)
    module.figures(curves,[],tmp_path,'campaign')
    assert len(checked)==1
