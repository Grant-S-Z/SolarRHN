import ROOT as rt
import math
import ctypes
import os
import numpy as np
from array import array

rt.gROOT.SetBatch(1)
rt.gStyle.SetOptFit(111)
rt.gStyle.SetOptStat(0)


def _edges_from_centers(centers, log_axis=False):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be a 1D array with at least 2 entries")

    edges = np.zeros(centers.size + 1, dtype=float)
    if log_axis:
        if np.any(centers <= 0):
            raise ValueError("centers must be > 0 for log axis")
        edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
        edges[0] = centers[0] * math.sqrt(centers[0] / centers[1])
        edges[-1] = centers[-1] * math.sqrt(centers[-1] / centers[-2])
    else:
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def rt_plot_exclusion_region(
    U2_values,
    MH_values,
    chi2_grid,
    file_name,
    dir='plots/',
    cl=0.90,
    ndof=2,
    xlog=True,
    ylog=False,
    type='pdf',
):
    """Plot 2D exclusion region from chi2 grid with ROOT.

    Excluded bins are defined as chi2 >= chi2_crit, where chi2_crit is
    computed from the requested CL and dof.

    Axis convention:
    - x-axis: MH (MeV)
    - y-axis: U2
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    U2_values = np.asarray(U2_values, dtype=float)
    MH_values = np.asarray(MH_values, dtype=float)
    chi2_grid = np.asarray(chi2_grid, dtype=float)

    if chi2_grid.shape != (U2_values.size, MH_values.size):
        raise ValueError(
            f"chi2_grid shape {chi2_grid.shape} does not match "
            f"(len(U2_values), len(MH_values))={(U2_values.size, MH_values.size)}"
        )

    x_edges = _edges_from_centers(MH_values, log_axis=xlog)
    y_edges = _edges_from_centers(U2_values, log_axis=ylog)

    h_chi2 = rt.TH2D(
        'h_chi2_excl',
        ';m_{N} (MeV);U^{2};#chi^{2}',
        len(MH_values),
        array('d', x_edges),
        len(U2_values),
        array('d', y_edges),
    )

    chi2_crit = rt.TMath.ChisquareQuantile(cl, ndof)
    print(f"Computed chi2 critical value for CL={cl}, dof={ndof}: {chi2_crit:.6f}")
    h_excl = rt.TH2D(
        'h_excl_region',
        f';m_{{N}} (MeV);U^{{2}};Excluded @ {int(100*cl)}% CL',
        len(MH_values),
        array('d', x_edges),
        len(U2_values),
        array('d', y_edges),
    )

    for ix in range(U2_values.size):
        for iy in range(MH_values.size):
            val = float(chi2_grid[ix, iy])
            # chi2_grid is indexed as [U2, MH], while histogram axes are [MH, U2].
            h_chi2.SetBinContent(iy + 1, ix + 1, val)
            h_excl.SetBinContent(iy + 1, ix + 1, 1.0 if val >= chi2_crit else 0.0)

    c1 = rt.TCanvas('c_excl', 'Exclusion region', 2200, 1600)
    c1.SetRightMargin(0.06)
    c1.SetLeftMargin(0.13)
    c1.SetBottomMargin(0.12)
    c1.SetTopMargin(0.08)
    if xlog:
        c1.SetLogx(1)
    if ylog:
        c1.SetLogy(1)

    h_excl.SetMinimum(0.0)
    h_excl.SetMaximum(1.0)
    h_excl.SetContour(2)
    # Two-color binary fill: 0 (allowed) and 1 (excluded), without color bar.
    palette = array('i', [rt.TColor.GetColor('#f7f7f7'), rt.TColor.GetColor('#fdae61')])
    rt.gStyle.SetPalette(2, palette)
    h_excl.Draw('col')

    # Overlay chi2 contour at the CL threshold.
    h_cont = h_chi2.Clone('h_chi2_contour')
    level = array('d', [chi2_crit])
    h_cont.SetContour(1, level)
    h_cont.SetLineColor(rt.kBlack)
    h_cont.SetLineWidth(3)
    h_cont.Draw('cont3 same')

    latex = rt.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.038)
    latex.DrawLatex(0.16, 0.94, f"{int(100*cl)}% CL exclusion, #chi^{{2}}_{{crit}}={chi2_crit:.3f} (dof={ndof})")

    out_path = os.path.join(dir, f"{file_name}.{type}")
    c1.SaveAs(out_path)
    return chi2_crit, out_path


def rt_plot_2d_heatmap(h2d, file_name, dir='plots/', n_levels=50, zmin=None, zmax=None, xlog=False, ylog=False, zlog=False, type='pdf'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    rt.gStyle.SetPalette(rt.kBlackBody)
    rt.TColor.InvertPalette()

    c1 = rt.TCanvas("c1", "hist2D", 2400, 1800)
    c1.SetRightMargin(0.18)
    c1.SetLeftMargin(0.13)
    c1.SetBottomMargin(0.13)
    c1.SetTopMargin(0.1)
    h2d.GetYaxis().SetTitleSize(0.055)
    h2d.GetYaxis().SetTitleOffset(1.1)
    h2d.GetYaxis().SetLabelSize(0.055)
    h2d.GetXaxis().SetTitleSize(0.055)
    h2d.GetXaxis().SetTitleOffset(1.0)
    h2d.GetXaxis().SetLabelSize(0.05)
    h2d.GetZaxis().SetTitleSize(0.055)
    h2d.GetZaxis().SetTitleOffset(1.1)
    h2d.GetZaxis().SetLabelSize(0.055)
    
    # h2d.SetContour(n_levels)
    h2d.GetZaxis().SetNdivisions(n_levels)
    # rt.gStyle.SetNumberContours(n_levels)

    if zmin is not None:
        h2d.SetMinimum(zmin)
    if zmax is not None:
        h2d.SetMaximum(zmax)
    if xlog:
        c1.SetLogx(1)
    if ylog:
        c1.SetLogy(1)
    if zlog:    
        c1.SetLogz(1)
    
    h2d.Draw("colz")
    c1.Update()
    c1.SaveAs(dir + file_name + "." + type)


def rt_plot_2d_contour(h2d, file_name, dir='plots/', n_levels=10, zmin=None, zmax=None, xlog=False, ylog=False, zlog=False, type='pdf'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    rt.gStyle.SetPalette(rt.kBlackBody)
    rt.TColor.InvertPalette()

    c1 = rt.TCanvas("c1", "hist2D", 2400, 1800)
    c1.SetRightMargin(0.18)
    c1.SetLeftMargin(0.13)
    c1.SetBottomMargin(0.13)
    c1.SetTopMargin(0.1)
    h2d.GetYaxis().SetTitleSize(0.055)
    h2d.GetYaxis().SetTitleOffset(1.1)
    h2d.GetYaxis().SetLabelSize(0.055)
    h2d.GetXaxis().SetTitleSize(0.055)
    h2d.GetXaxis().SetTitleOffset(1.0)
    h2d.GetXaxis().SetLabelSize(0.05)
    h2d.GetZaxis().SetTitleSize(0.055)
    h2d.GetZaxis().SetTitleOffset(1.1)
    h2d.GetZaxis().SetLabelSize(0.055)
    
    h2d.SetContour(n_levels)
    h2d.GetZaxis().SetNdivisions(n_levels)
    # rt.gStyle.SetNumberContours(n_levels)

    if zmin is not None:
        h2d.SetMinimum(zmin)
    if zmax is not None:
        h2d.SetMaximum(zmax)
    if xlog:
        c1.SetLogx(1)
    if ylog:
        c1.SetLogy(1)
    if zlog:    
        c1.SetLogz(1)
    
    h2d.Draw("colz")
    c1.Update()

    h2d_cont = h2d.Clone()
    h2d_cont.SetLineColor(rt.kBlack)
    h2d_cont.Draw("cont3 same")
    c1.SaveAs(dir + file_name + "." + type)


# def plot2DContour(h2d, file_name, dir='plots/', zmin=None, zmax=None, contours=None, xlog=False, ylog=False, zlog=False, type='pdf'):
#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     rt.gStyle.SetPalette(rt.kBlackBody)
#     rt.TColor.InvertPalette()

#     canvas = rt.TCanvas("canvas", "hist2D", 2400, 1800)
#     canvas.SetRightMargin(0.18)
#     canvas.SetLeftMargin(0.13)
#     canvas.SetBottomMargin(0.13)
#     canvas.SetTopMargin(0.1)

#     h2d.GetYaxis().SetTitleSize(0.055)
#     h2d.GetYaxis().SetTitleOffset(1.1)
#     h2d.GetYaxis().SetLabelSize(0.055)
#     h2d.GetXaxis().SetTitleSize(0.055)
#     h2d.GetXaxis().SetTitleOffset(1.0)
#     h2d.GetXaxis().SetLabelSize(0.05)
#     h2d.GetZaxis().SetTitleSize(0.055)
#     h2d.GetZaxis().SetTitleOffset(1.1)
#     h2d.GetZaxis().SetLabelSize(0.055)
#     if zmin is not None:
#         h2d.SetMinimum(zmin)
#         h2d.SetMaximum(zmax)
#     h2d.Draw("colz")

#     # canvas.SetLogx(0)
#     # canvas.SetLogz(0)
#     # canvas.SetLogy(0)
#     if xlog:
#         canvas.SetLogx(1)
#     if ylog:
#         canvas.SetLogy(1)
#     if zlog:    
#         canvas.SetLogz(1)
#     canvas.SaveAs(dir + file_name + "." + type)

#     if contours is None:
#         return

#     h2d.SetContour(len(contours), contours)
#     h2d.Draw("CONT Z LIST")
#     canvas.Update()
#     contours_list = rt.gROOT.GetListOfSpecials().FindObject("contours")

#     h2d.SetLineColor(rt.kBlack)
#     h2d.Draw("colz")
#     canvas.Update()
#     h2d.Draw("CONT3 SAME")

#     if contours_list:
#         print("contours_list")
#         text = rt.TLatex()
#         text.SetTextSize(0.03)
#         for i in range(contours_list.GetSize()):
#             if i + 2 > len(contours):
#                 continue
#             contour_list = contours_list.At(i)
#             for contour in contour_list:
#                 graph = contour.Clone()
#                 n_points = graph.GetN()
#                 if n_points > 2:
#                     x_center_ref = ctypes.c_double(0)
#                     y_center_ref = ctypes.c_double(0)
#                     n_center = int(n_points / 2)
#                     graph.GetPoint(n_center, x_center_ref, y_center_ref)
#                     x_center = x_center_ref.value
#                     y_center_log = y_center_ref.value
#                     y_center = pow(10, y_center_log)

#                     text_show = str(contours[i + 1])
#                     if zlog:
#                         pow10 = int(math.log10(contours[i + 1]))
#                         text_show = "10^{" + str(pow10) + "} s"
#                     # Draw the contour value at the center of the contour line
#                     text.DrawLatex(x_center, y_center, text_show)

#     canvas.Update()

#     canvas.SaveAs(dir + file_name + "_contour." + type)


def rt_plot_exclusion_from_csv(
    csv_path,
    file_name,
    dir="plots/",
    drawborexino=False,
    borexino_csv="./data/Borexino_exclusion.csv",
    cl=0.90,
    smooth=True,
    smooth_samples=2000,
    type="pdf",
):
    """Read exclusion CSV (mH, u2_low, u2_high) and plot with ROOT.

    CSV columns: mH, u2_low, u2_high
    - no-limit points have u2_low ≈ u2_high ≥ U2_MAX
    - window points have u2_low < u2_high < U2_MAX
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    mh = data[:, 0]
    u2_low = data[:, 1]
    u2_high = data[:, 2]

    # Heuristic: no-limit points have u2 values near the max
    y_max = np.max(u2_low) * 0.8
    valid = u2_low < y_max
    window = valid & (u2_high < y_max)

    # ── TGraph for lower boundary ──
    n_low = int(np.sum(valid))
    gr_low = rt.TGraph(n_low)
    j = 0
    for i in range(len(mh)):
        if valid[i]:
            gr_low.SetPoint(j, float(mh[i]), float(u2_low[i]))
            j += 1
    gr_low.SetLineColor(rt.kAzure + 2)
    gr_low.SetLineWidth(3)

    # ── TGraph for upper boundary (window points) ──
    n_win = int(np.sum(window))
    gr_high = None
    if n_win > 0:
        gr_high = rt.TGraph(n_win)
        j = 0
        for i in range(len(mh)):
            if window[i]:
                gr_high.SetPoint(j, float(mh[i]), float(u2_high[i]))
                j += 1
        gr_high.SetLineColor(rt.kAzure + 2)
        gr_high.SetLineWidth(3)

    # ── Canvas ──
    c = rt.TCanvas("c_excl_csv", "Expected exclusion", 1600, 1200)
    c.SetRightMargin(0.06)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.08)
    c.SetLogy(1)

    y_min = 1e-6
    y_frame = 1e-1
    frame = c.DrawFrame(mh[0], y_min, mh[-1], y_frame)
    frame.GetXaxis().SetTitle("m_{H}  [MeV]")
    frame.GetYaxis().SetTitle("|U_{eH}|^{2}")
    frame.GetYaxis().SetTitleOffset(1.2)
    frame.SetTitle("")

    def _smooth_xy_with_spline(x_vals, y_vals, spline_name, min_samples=2000, log_y=False):
        """Return spline object (if possible) and spline-sampled (x, y)."""
        n_pts = len(x_vals)
        pairs = sorted(zip([float(x) for x in x_vals], [float(y) for y in y_vals]), key=lambda t: t[0])
        x_u, y_u = [], []
        for x, y in pairs:
            if x_u and abs(x - x_u[-1]) < 1e-12:
                y_u[-1] = y
            else:
                x_u.append(x)
                y_u.append(y)

        if len(x_u) < 3:
            return None, x_u, y_u

        gr_tmp = rt.TGraph(len(x_u))
        for idx, (xv, yv) in enumerate(zip(x_u, y_u)):
            y_safe = max(float(yv), 1e-300)
            y_in = float(np.log10(y_safe)) if log_y else y_safe
            gr_tmp.SetPoint(idx, float(xv), y_in)

        spl = rt.TSpline3(spline_name, gr_tmp)
        spl.SetNpx(max(min_samples, len(x_u) * 40))

        x_dense = np.linspace(float(x_u[0]), float(x_u[-1]), spl.GetNpx())
        y_dense = []
        for xv in x_dense:
            sval = float(spl.Eval(float(xv)))
            y_dense.append(float(10.0**sval) if log_y else sval)
        return spl, [float(xv) for xv in x_dense], y_dense

    # Prepare boundary points on valid mass range
    x_valid = [float(mh[i]) for i in range(len(mh)) if valid[i]]
    y_low_valid = [float(u2_low[i]) for i in range(len(mh)) if valid[i]]
    y_top_valid = [float(u2_high[i]) if window[i] else y_frame for i in range(len(mh)) if valid[i]]

    # Build boundaries for plotting/fill
    low_spl = None
    x_low_plot, y_low_plot = x_valid, y_low_valid
    if n_low > 0 and smooth:
        low_spl, x_low_plot, y_low_plot = _smooth_xy_with_spline(
            x_valid, y_low_valid, "spl_low", min_samples=smooth_samples, log_y=True
        )

    # Upper boundary for fill (includes y_frame on no-window points)
    top_fill_spl = None
    x_top_fill, y_top_fill = x_valid, y_top_valid
    if n_low > 0 and smooth:
        top_fill_spl, x_top_fill, y_top_fill = _smooth_xy_with_spline(
            x_valid, y_top_valid, "spl_top_fill", min_samples=smooth_samples, log_y=True
        )

    # Keep boundaries physical/stable on log-y axis
    y_floor = y_min * 0.5
    y_low_plot = [max(y_floor, float(yv)) for yv in y_low_plot]
    y_top_fill = [
        max(y_low_plot[i], min(y_frame, max(y_floor, float(y_top_fill[i]))))
        for i in range(len(y_top_fill))
    ]

    # ── Excluded fill between lower and upper (smoothed) ──
    if n_low > 0:
        x_fill = list(x_low_plot)
        y_fill = list(y_low_plot)
        for i in range(len(x_top_fill) - 1, -1, -1):
            x_fill.append(float(x_top_fill[i]))
            y_fill.append(float(y_top_fill[i]))

        n_fill = len(x_fill)
        gr_fill = rt.TGraph(n_fill)
        for i in range(n_fill):
            gr_fill.SetPoint(i, x_fill[i], y_fill[i])
        gr_fill.SetFillColorAlpha(rt.kAzure + 2, 0.20)
        gr_fill.SetLineWidth(0)
        gr_fill.Draw("F same")

    # ── Boundary lines ──
    if n_low > 0:
        gr_low_plot = rt.TGraph(len(x_low_plot))
        for i in range(len(x_low_plot)):
            gr_low_plot.SetPoint(i, x_low_plot[i], y_low_plot[i])
        gr_low_plot.SetLineColor(rt.kAzure + 2)
        gr_low_plot.SetLineWidth(3)
        gr_low_plot.SetMarkerSize(0)
        gr_low_plot.Draw("C same" if smooth else "L same")

    if gr_high:
        x_win = [float(mh[i]) for i in range(len(mh)) if window[i]]
        y_win = [float(u2_high[i]) for i in range(len(mh)) if window[i]]
        x_high_plot, y_high_plot = x_win, y_win
        if smooth:
            _, x_high_plot, y_high_plot = _smooth_xy_with_spline(
                x_win, y_win, "spl_high", min_samples=smooth_samples, log_y=True
            )
        y_high_plot = [max(y_floor, min(y_frame, float(yv))) for yv in y_high_plot]

        gr_high_plot = rt.TGraph(len(x_high_plot))
        for i in range(len(x_high_plot)):
            gr_high_plot.SetPoint(i, x_high_plot[i], y_high_plot[i])
        gr_high_plot.SetLineColor(rt.kAzure + 2)
        gr_high_plot.SetLineWidth(3)
        gr_high_plot.SetMarkerSize(0)
        gr_high_plot.Draw("C same" if smooth else "L same")

    # Connect the two right-most points at the same x (close the boundary on the right)
    if n_low > 0 and len(x_top_fill) > 0:
        x_right = float(x_low_plot[-1])
        y_right_low = float(y_low_plot[-1])
        y_right_high = float(y_top_fill[-1])
        gr_right_close = rt.TGraph(2)
        gr_right_close.SetPoint(0, x_right, y_right_low)
        gr_right_close.SetPoint(1, x_right, y_right_high)
        gr_right_close.SetLineColor(rt.kAzure + 2)
        gr_right_close.SetLineWidth(3)
        gr_right_close.SetMarkerSize(0)
        gr_right_close.Draw("L same")

    # ── Borexino published exclusion ──
    if drawborexino:
        gr_ref = None
        if os.path.exists(borexino_csv):
            ref = np.loadtxt(borexino_csv, delimiter=",")
            if ref.ndim == 2 and ref.shape[1] >= 2:
                log10_mh, log10_u2 = ref[:, 0], ref[:, 1]
                mh_ref = (10.0**log10_mh) * 1e3
                u2_ref = 10.0**log10_u2
                n_ref = len(mh_ref)
                gr_ref = rt.TGraph(n_ref)
                for i in range(n_ref):
                    gr_ref.SetPoint(i, float(mh_ref[i]), float(u2_ref[i]))
                    gr_ref.SetLineColor(rt.kGray + 2)
                    gr_ref.SetLineWidth(2)
                    gr_ref.SetLineStyle(7)
                    gr_ref.Draw("L same")


    # ── Legend ──
    leg = rt.TLegend(0.65, 0.70, 0.85, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(gr_low, f"{file_name}", "l")
    if drawborexino and gr_ref:
        leg.AddEntry(gr_ref, "Borexino (published)", "l")
    leg.Draw()

    out_path = os.path.join(dir, f"{file_name}.{type}")
    c.SaveAs(out_path)
    print(f"Saved: {out_path}")
    return out_path
