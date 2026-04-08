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