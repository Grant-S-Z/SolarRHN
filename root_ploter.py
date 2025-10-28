import numpy as np
import ROOT as rt
import math
import ctypes
from utils import *
from sampling import *
import threading
import time
import multiprocessing
from multiprocessing import sharedctypes

rt.gROOT.SetBatch(1)
rt.gStyle.SetOptFit(111)
rt.gStyle.SetOptStat(0)


def plot2DContour(h2d, fileName, zmin=None, zmax=None, contours=None, useLog10=False):
    rt.gStyle.SetPalette(rt.kBlackBody)
    rt.TColor.InvertPalette()

    canvas = rt.TCanvas("canvas", "hist2D", 2400, 1800)
    canvas.SetRightMargin(0.18)
    canvas.SetLeftMargin(0.13)
    canvas.SetBottomMargin(0.13)
    canvas.SetTopMargin(0.1)

    canvas.SetLogx(0)
    canvas.SetLogy(1)
    canvas.SetLogz(1)

    h2d.GetYaxis().SetTitleSize(0.055)
    h2d.GetYaxis().SetTitleOffset(1.1)
    h2d.GetYaxis().SetLabelSize(0.055)
    h2d.GetXaxis().SetTitleSize(0.055)
    h2d.GetXaxis().SetTitleOffset(1.0)
    h2d.GetXaxis().SetLabelSize(0.05)
    h2d.GetZaxis().SetTitleSize(0.055)
    h2d.GetZaxis().SetTitleOffset(1.1)
    h2d.GetZaxis().SetLabelSize(0.055)
    if zmin is not None:
        h2d.SetMinimum(zmin)
        h2d.SetMaximum(zmax)
    h2d.Draw("colz")

    canvas.SaveAs(fileName + "_linXlogYlogZ.pdf")
    canvas.SaveAs(fileName + "_linXlogYlogZ.png")

    canvas.SetLogx(1)
    canvas.SaveAs(fileName + "_logXlogYlogZ.pdf")
    canvas.SaveAs(fileName + "_logXlogYlogZ.png")

    canvas.SetLogx(0)
    canvas.SetLogz(0)
    canvas.SaveAs(fileName + "_linXlogYlinZ.pdf")
    canvas.SaveAs(fileName + "_linXlogYlinZ.png")

    canvas.SetLogy(0)
    canvas.SaveAs(fileName + "_linXlinYlinZ.pdf")
    canvas.SaveAs(fileName + "_linXlinYlinZ.png")
    canvas.SetLogz(1)
    canvas.SaveAs(fileName + "_linXlinYlogZ.pdf")
    canvas.SaveAs(fileName + "_linXlinYlogZ.png")

    canvas.SetLogy(0)
    if contours is None:
        return

    canvas.SetLogx(0)
    canvas.SetLogy(1)
    canvas.SetLogz(1)
    h2d.SetContour(len(contours), contours)
    h2d.Draw("CONT Z LIST")
    canvas.Update()
    contours_list = rt.gROOT.GetListOfSpecials().FindObject("contours")

    h2d.SetLineColor(rt.kBlack)
    h2d.Draw("colz")
    canvas.Update()
    h2d.Draw("CONT3 SAME")

    if contours_list:
        print("contours_list")
        text = rt.TLatex()
        text.SetTextSize(0.03)
        for i in range(contours_list.GetSize()):
            if i + 2 > len(contours):
                continue
            contour_list = contours_list.At(i)
            for contour in contour_list:
                graph = contour.Clone()
                n_points = graph.GetN()
                if n_points > 2:
                    x_center_ref = ctypes.c_double(0)
                    y_center_ref = ctypes.c_double(0)
                    n_center = int(n_points / 2)
                    graph.GetPoint(n_center, x_center_ref, y_center_ref)
                    x_center = x_center_ref.value
                    y_center_log = y_center_ref.value
                    y_center = pow(10, y_center_log)

                    text_show = str(contours[i + 1])
                    if useLog10:
                        pow10 = int(math.log10(contours[i + 1]))
                        text_show = "10^{" + str(pow10) + "} s"
                    # Draw the contour value at the center of the contour line
                    text.DrawLatex(x_center, y_center, text_show)

    canvas.Update()

    canvas.SaveAs(fileName + "_contour_linXlogYlogZ.pdf")
    canvas.SaveAs(fileName + "_contour_linXlogYlogZ.png")


def plotBR_vsMU(Umin, Umax, nU, Mmin, Mmax, nM):
    logU_min = math.log10(Umin)
    logU_max = math.log10(Umax)
    logU_step = (logU_max - logU_min) / nU
    ybins = []
    for iU in range(nU):
        ybins.append(pow(10, logU_min + logU_step * iU))
    ybins.append(pow(10, logU_min + logU_step * nU))

    h2d = rt.TH2D(
        "hist",
        ";m_{#nuH} (MeV);U_{eH}^{2};BR_{#null}",
        nM,
        Mmin,
        Mmax,
        nU,
        np.array(ybins, dtype=np.float64),
    )

    for iU in range(nU):
        U_this = h2d.GetYaxis().GetBinCenter(iU + 1)
        for iM in range(nM):
            M_this = h2d.GetXaxis().GetBinCenter(iM + 1)
            BR_this = RHN_BR_vll(M_this, U_this)
            h2d.SetBinContent(iM + 1, iU + 1, BR_this)
    # contours_tau = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], dtype=np.float64)

    # plot2DContour(h2d, "plots/RHNBRvll_vs_M_U2", zmin=1e-1, zmax=1e10, contours = contours_tau, useLog10=True)
    plot2DContour(h2d, "plots/RHNBRvll_vs_M_U2")


def plotTauCM_vsMU(Umin, Umax, nU, Mmin, Mmax, nM):
    logU_min = math.log10(Umin)
    logU_max = math.log10(Umax)
    logU_step = (logU_max - logU_min) / nU
    ybins = []
    for iU in range(nU):
        ybins.append(pow(10, logU_min + logU_step * iU))
    ybins.append(pow(10, logU_min + logU_step * nU))

    h2d = rt.TH2D(
        "hist",
        ";m_{#nuH} (MeV);U_{eH}^{2};#tau_{c.m.} (s)",
        nM,
        Mmin,
        Mmax,
        nU,
        np.array(ybins, dtype=np.float64),
    )

    for iU in range(nU):
        U_this = h2d.GetYaxis().GetBinCenter(iU + 1)
        for iM in range(nM):
            M_this = h2d.GetXaxis().GetBinCenter(iM + 1)
            tau_this = RHN_TauCM(M_this, U_this)
            h2d.SetBinContent(iM + 1, iU + 1, tau_this)
    contours_tau = np.array(
        [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], dtype=np.float64
    )

    plot2DContour(
        h2d,
        "plots/RHNTauCM_vs_M_U2",
        zmin=1e-1,
        zmax=1e10,
        contours=contours_tau,
        useLog10=True,
    )


def plotCount(Umin, Umax, nU, Mmin, Mmax, nM):
    logU_min = math.log10(Umin)
    logU_max = math.log10(Umax)
    logU_step = (logU_max - logU_min) / nU
    ybins = []
    for iU in range(nU):
        ybins.append(pow(10, logU_min + logU_step * iU))
    ybins.append(pow(10, logU_min + logU_step * nU))

    h2d = rt.TH2D(
        "hist",
        ";m_{#nuH} (MeV);U_{eH}^{2};Counts",
        nM,
        Mmin,
        Mmax,
        nU,
        np.array(ybins, dtype=np.float64),
    )

    for iU in range(nU):
        U_this = h2d.GetYaxis().GetBinCenter(iU + 1)
        for iM in range(nM):
            M_this = h2d.GetXaxis().GetBinCenter(iM + 1)
            tau_this = RHN_TauCM(M_this, U_this)
            h2d.SetBinContent(iM + 1, iU + 1, tau_this)
    contours_tau = np.array(
        [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], dtype=np.float64
    )

    plot2DContour(
        h2d,
        "plots/RHNTauCM_vs_M_U2",
        zmin=1e-1,
        zmax=1e10,
        contours=contours_tau,
        useLog10=True,
    )


def plotDiff_vs_E_costheta(spectrum_L, MH, U2):
    energy = spectrum_L[:, 0]

    npoints_costheta = 201
    costheta_step = 2.0 / (npoints_costheta - 1.0)
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * costheta_step

    spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
    flux_R = spectrum_R[:, 1]

    diff_El_costheta = np.zeros((len(energy), npoints_costheta))
    for ieH in range(len(energy)):
        weight = flux_R[ieH]
        if weight < 1e-6:
            continue
        EH = energy[ieH]
        diff_El_costheta_temp = np.zeros((len(energy), npoints_costheta))
        for ieL in range(len(energy)):
            for iTh in range(npoints_costheta):
                diff_El_costheta_temp[ieL][iTh] = diff_El_costheta_lab(
                    energy[ieL], costheta_arr[iTh], MH, EH
                )
        sum_temp = np.sum(diff_El_costheta_temp)
        if sum_temp > 0.0:
            for ieL in range(len(energy)):
                for iTh in range(npoints_costheta):
                    diff_El_costheta[ieL][iTh] += (
                        weight * diff_El_costheta_temp[ieL][iTh] / sum_temp
                    )

    sum_diff_all = np.sum(diff_El_costheta)
    # print(diff_El_costheta)

    xbins = np.append(energy, [2.0 * energy[-1] - energy[-2]])
    ybins = np.append(costheta_arr, [2.0 * costheta_arr[-1] - costheta_arr[-2]])

    h2d = rt.TH2D(
        "hist",
        "M_{#nu_{H}} = "
        + str(MH)
        + " MeV, U^{2} = "
        + str(U2)
        + ";Neutrino #nu_{e} Energy (MeV);Emission Angle cos(#theta);r.u.",
        len(energy),
        np.array(xbins, dtype=np.float64),
        npoints_costheta,
        np.array(ybins, dtype=np.float64),
    )

    for ieL in range(len(energy)):
        for iTh in range(npoints_costheta):
            h2d.SetBinContent(
                ieL + 1, iTh + 1, diff_El_costheta[ieL][iTh] / sum_diff_all
            )
    contours_tau = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0], dtype=np.float64)

    plot2DContour(h2d, "plots/Diff_vs_El_costheta_M" + str(MH) + "_U" + str(U2))


# plot a 1D Curve, input is a 2D numpy array
def plot1DCurve(
    arr2D,
    plotTitle,
    xtitle,
    ytitle,
    fileName,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    if canvas is None:
        canvas = rt.TCanvas("canvas", "arr2D", 800, 600)
        canvas.SetRightMargin(0.02)
        canvas.SetLeftMargin(0.13)
        canvas.SetBottomMargin(0.13)
        canvas.SetTopMargin(0.1)

    canvas.SetLogx(0)
    canvas.SetLogy(0)
    xvalues = arr2D[:, 0]
    yvalues = arr2D[:, 1]

    xmin2 = min(xvalues) * 0.9
    xmax2 = max(xvalues) * 1.1
    ymin2 = min(yvalues) * 0.9
    ymax2 = max(yvalues) * 1.2

    if xmin is None:
        xmin = xmin2
        xmax = xmax2
        ymin = ymin2
        ymax = ymax2
    # if ymin > ymin2 and ymin2 > 0.0:
    #    ymin = ymin2
    if ymax < ymax2:
        ymax = ymax2

    graph = rt.TGraph(len(arr2D), xvalues.astype(float), yvalues.astype(float))

    graph.Draw("AL")
    graph.SetLineWidth(2)
    graph.GetYaxis().SetTitleSize(0.055)
    graph.GetYaxis().SetTitleOffset(1.2)
    graph.GetYaxis().SetLabelSize(0.055)
    graph.GetYaxis().SetRangeUser(0, ymax)
    graph.GetXaxis().SetTitleSize(0.055)
    graph.GetXaxis().SetTitleOffset(1.0)
    graph.GetXaxis().SetLabelSize(0.05)
    if xmin > 0.0:
        graph.GetXaxis().SetLimits(0.0, xmax)
    else:
        graph.GetXaxis().SetLimits(xmin, xmax)

    graph.SetTitle(plotTitle + ";" + xtitle + ";" + ytitle)

    canvas.SaveAs(fileName + "_linXlinY.pdf")
    canvas.SaveAs(fileName + "_linXlinY.png")

    canvas.SetLogx(0)
    canvas.SetLogy(1)
    if xmin > 0.0:
        graph.GetXaxis().SetLimits(0.0, xmax)
    else:
        graph.GetXaxis().SetLimits(xmin, xmax)
    graph.GetYaxis().SetRangeUser(ymin, ymax * 1000)
    canvas.SaveAs(fileName + "_linXlogY.pdf")
    canvas.SaveAs(fileName + "_linXlogY.png")

    if xmin > 0.0:
        canvas.SetLogx(1)
        canvas.SetLogy(1)
        graph.GetXaxis().SetLimits(xmin, xmax)
        graph.GetYaxis().SetRangeUser(ymin, ymax * 100)
        canvas.SaveAs(fileName + "_logXlogY.pdf")
        canvas.SaveAs(fileName + "_logXlogY.png")


def plotSpectrums(
    spectrums,
    titles,
    xtitle,
    ytitle,
    fileName,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
    labels=None,
    column_names="energy,flux",
    legNColumns=3,
):
    if canvas is None:
        canvas = rt.TCanvas("canvas", "spectrum", 800, 600)
        canvas.SetRightMargin(0.02)
        canvas.SetLeftMargin(0.15)
        canvas.SetBottomMargin(0.13)
        canvas.SetTopMargin(0.08)

    if labels is not None:
        saveSpectrums(spectrums, column_names, fileName, labels)

    canvas.SetLogx(0)
    canvas.SetLogy(0)

    graphs = []
    xmin2 = min(spectrums[0][:, 0]) * 0.9
    xmax2 = max(spectrums[0][:, 0]) * 1.1

    ymin2 = min(np.vstack(spectrums)[:, 1]) * 0.9
    ymax2 = max(np.vstack(spectrums)[:, 1]) * 1.5

    if xmin is None:
        xmin = xmin2
        xmax = xmax2
        ymin = ymin2
        ymax = ymax2

    # if ymin > ymin2 and ymin2 > 0.0:
    #    ymin = ymin2
    if ymax < ymax2:
        ymax = ymax2

    for isp in range(len(spectrums)):
        spectrum = spectrums[isp]
        xvalues = spectrum[:, 0]
        yvalues = spectrum[:, 1]

        graph = rt.TGraph(len(spectrum), xvalues.astype(float), yvalues.astype(float))
        graph.SetLineWidth(2)
        graph.SetLineColor(isp + 1)
        graph.SetMarkerColor(isp + 1)
        graphs.append(graph)

    graphs[0].Draw("AL")
    graphs[0].GetYaxis().SetTitleSize(0.055)
    graphs[0].GetYaxis().SetTitleOffset(1.4)
    graphs[0].GetYaxis().SetLabelSize(0.055)
    graphs[0].GetYaxis().SetRangeUser(0, ymax)
    graphs[0].GetXaxis().SetTitleSize(0.055)
    graphs[0].GetXaxis().SetTitleOffset(1.0)
    graphs[0].GetXaxis().SetLabelSize(0.05)
    if xmin > 0.0:
        graphs[0].GetXaxis().SetLimits(0.0, xmax)
    else:
        graphs[0].GetXaxis().SetLimits(xmin, xmax)
    graphs[0].SetTitle(";" + xtitle + ";" + ytitle)
    if len(titles) > len(spectrums):
        graphs[0].SetTitle(titles[-1] + ";" + xtitle + ";" + ytitle)

    leg = rt.TLegend(0.16, 0.73, 0.97, 0.92)
    leg.SetNColumns(legNColumns)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.032)

    for ig in range(len(graphs)):
        if ig > 0:
            graphs[ig].Draw("L same")
        leg.AddEntry(graphs[ig], titles[ig], "l")

    leg.Draw()

    canvas.SaveAs(fileName + "_linXlinY.pdf")
    canvas.SaveAs(fileName + "_linXlinY.png")

    canvas.SetLogx(0)
    canvas.SetLogy(1)
    if xmin > 0.0:
        graphs[0].GetXaxis().SetLimits(0.0, xmax)
    else:
        graphs[0].GetXaxis().SetLimits(xmin, xmax)
    graphs[0].GetYaxis().SetRangeUser(ymin, ymax * 100)  # multiply one factor
    canvas.SaveAs(fileName + "_linXlogY.pdf")
    canvas.SaveAs(fileName + "_linXlogY.png")

    if xmin > 0.0:
        canvas.SetLogx(1)
        canvas.SetLogy(1)
        graphs[0].GetYaxis().SetRangeUser(ymin, ymax * 100)
        graphs[0].GetXaxis().SetLimits(xmin, xmax)
        canvas.SaveAs(fileName + "_logXlogY.pdf")
        canvas.SaveAs(fileName + "_logXlogY.png")


def plotSpectrums_grid1d(
    spectrums1d,
    spectrum_nuL,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    labels = []
    titles = []

    for ig in range(len(grid1d_U2M)):
        labels.append("U" + str(grid1d_U2M[ig][0]) + "_M" + str(grid1d_U2M[ig][1]))
        titles.append(
            "U^{2} = "
            + str(int(grid1d_U2M[ig][0]))
            + ", m_{#nuH} = "
            + str(int(grid1d_U2M[ig][1]))
            + " MeV"
        )
        plot1DCurve(
            spectrums1d[ig],
            titles[ig],
            xtitle,
            ytitle,
            fileName + "_" + labels[ig],
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
    U = grid1d_U2M[0][0]
    spectrums1d.insert(0, spectrum_nuL)
    titles.insert(0, "#nu_{e}")
    labels.insert(0, "nuL")
    plotSpectrums(
        spectrums1d,
        titles,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels,
        "energy,flux",
    )


def plot_nuL_El_grid1d(
    spectrum_L,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    energy = spectrum_L[:, 0]  # energy points

    labels_all = []
    titles_all = []

    spectrums_all = []

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append(
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV"
        )

        spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
        flux_R = spectrum_R[:, 1]  # spectrum of nuH

        spectrum_nuL = np.zeros(
            (len(energy), 2)
        )  # spectrum of left-handed neutrino after nuH decay
        for ieL in range(len(energy)):
            spectrum_nuL[ieL][0] = energy[ieL]

        for ieH in range(len(energy)):
            weight = flux_R[ieH]
            if weight < 1e-6:
                continue
            diff_ieL_temp = np.zeros(len(energy))
            for ieL in range(len(energy)):
                diff_ieL_temp[ieL] = diff_El(energy[ieL], MH, energy[ieH])
            sum_diff_ieL_temp = np.sum(diff_ieL_temp)
            for ieL in range(len(energy)):
                spectrum_nuL[ieL][1] += weight * diff_ieL_temp[ieL] / sum_diff_ieL_temp

        spectrums_3types = []
        titles = [
            "#nu_{e} from ^{8}B",
            "#nu_{H} (U^{2}=" + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV)",
            "#nu_{e} from #nu_{H}",
        ]
        spectrums_3types.append(spectrum_L)
        spectrums_3types.append(spectrum_R)
        spectrums_3types.append(spectrum_nuL)

        plotSpectrums(
            spectrums_3types,
            titles,
            "Neutrino Energy (MeV)",
            ytitle,
            fileName + "_U" + str(U2) + "_M" + str(MH),
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
        spectrums_all.append(spectrum_nuL)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]
    spectrums_all.insert(0, spectrum_L)
    titles_all.insert(0, "#nu_{e} from ^{8}B")
    labels_all.insert(0, "nuL")
    plotSpectrums(
        spectrums_all,
        titles_all,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels_all,
        "energy,flux",
    )


def plot_nuL_costheta_grid1d(
    spectrum_L,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 203
    costheta_step = 0.01
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.01 + i * costheta_step
    costheta_arr[-2] = 1.0

    labels_all = []
    titles_all = []

    diff_costheta_all = []

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append(
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV"
        )

        spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
        flux_R = spectrum_R[:, 1]  # spectrum of nuH

        diff_costheta_this = np.zeros((npoints_costheta, 2))

        for icostheta in range(npoints_costheta):
            diff_costheta_this[icostheta][0] = costheta_arr[icostheta]

        for ieH in range(len(energy)):
            weight = flux_R[ieH]
            if weight < 1e-6:
                continue
            diff_costheta_temp = np.zeros(npoints_costheta)
            for icostheta in range(npoints_costheta):
                diff_costheta_temp[icostheta] = diff_costheta(
                    costheta_arr[icostheta],
                    MH,
                    energy[ieH],
                )
            sum_diff_costheta_temp = np.sum(diff_costheta_temp)
            for icostheta in range(npoints_costheta):
                diff_costheta_this[icostheta][1] += (
                    weight * diff_costheta_temp[icostheta] / sum_diff_costheta_temp
                )

        # normalize to 1
        sum_diff_costheta = np.sum(diff_costheta_this[:, 1])
        for icostheta in range(npoints_costheta):
            diff_costheta_this[icostheta][1] *= (
                (npoints_costheta - 1) * 1.0 / sum_diff_costheta
            )

        plot1DCurve(
            diff_costheta_this,
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV",
            xtitle,
            ytitle,
            fileName + "_U" + str(U2) + "_M" + str(MH),
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
        diff_costheta_all.append(diff_costheta_this)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]

    plotSpectrums(
        diff_costheta_all,
        titles_all,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels_all,
        "costheta,flux",
    )


# detector_S: cm^2
# detector_L: m
# exposure_time: s
def plot_Eee_in_detector_grid1d(
    spectrum_L,
    background_data,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    detector_S,
    detector_L,
    exposure_time,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 201
    costheta_step = 2.0 / (npoints_costheta - 1.0)
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * costheta_step

    labels_all = []
    titles_all = []

    spectrums_all = []

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append("m_{#nuH} = " + str(int(MH)) + " MeV")

        spectrum_R_orig = getRHNSpectrum(spectrum_L, MH, U2)
        spectrum_R = getDecayedRHNSpectrum(
            spectrum_R_orig, MH, U2, distance_SE, detector_L
        )
        for ie in range(len(spectrum_R)):
            spectrum_R[ie][1] = spectrum_R[ie][1] * detector_S * exposure_time

        flux_R = spectrum_R[:, 1]  # spectrum of nuH

        spectrum_nuL = np.zeros(
            (len(energy), 2)
        )  # spectrum of left-handed neutrino after nuH decay
        spectrum_Eee = np.zeros(
            (len(energy), 2)
        )  # spectrum of electron pair after nuH decay
        spectrum_CosthetaL = np.zeros(
            (len(costheta_arr), 2)
        ) # spectrum of costheta of electron after nuH decay
        for ieL in range(len(energy)):
            spectrum_nuL[ieL][0] = energy[ieL]
            spectrum_Eee[ieL][0] = energy[ieL]

        for ieH in range(len(energy)):
            weight = flux_R[ieH]
            if weight < 1e-6:
                continue
            diff_ieL_temp = np.zeros(len(energy))
            diff_iEee_temp = np.zeros(len(energy))
            diff_iCosthetaL_temp = np.zeros(len(energy))

            for ieL in range(len(energy)):
                diff_ieL_temp[ieL] = diff_El(energy[ieL], MH, energy[ieH])
                if energy[ieH] - energy[ieL] > 0.0:
                    diff_iEee_temp[ieL] = diff_El(
                        energy[ieH] - energy[ieL], MH, energy[ieH]
                    )

            sum_diff_ieL_temp = np.sum(diff_ieL_temp)
            sum_diff_iEee_temp = np.sum(diff_iEee_temp)

            for iCosthetaL in range(len(costheta_arr)):
                diff_iCosthetaL_temp[iCosthetaL] = diff_costheta(costheta_arr[iCosthetaL], MH, energy[ieH])

            sum_diff_iCosthetaL_temp = np.sum(diff_iCosthetaL_temp)

            for ieL in range(len(energy)):
                spectrum_nuL[ieL][1] += weight * diff_ieL_temp[ieL] / sum_diff_ieL_temp
                spectrum_Eee[ieL][1] += (
                    weight * diff_iEee_temp[ieL] / sum_diff_iEee_temp
                )

            for iCosthetaL in range(len(costheta_arr)):
                spectrum_CosthetaL += weight * diff_iCosthetaL_temp[iCosthetaL] / sum_diff_iCosthetaL_temp

        spectrums_4types = []
        titles = [
            "#nu_{H} (U^{2}=" + str(U2) + ", m_{#nuH} = " + str(int(MH)) + " MeV)",
            "#nu_{e} from #nu_{H}",
            "e^{+}e^{-} from #nu_{H}",
        ]
        # spectrums_4types.append(spectrum_L)
        spectrums_4types.append(spectrum_R)
        spectrums_4types.append(spectrum_nuL)
        spectrums_4types.append(spectrum_Eee)
        spectrums_4types.append(spectrum_CosthetaL)

        plotSpectrums(
            spectrums_4types,
            titles,
            "Energy (MeV)",
            ytitle,
            fileName + "_U" + str(U2) + "_M" + str(MH),
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
        spectrums_all.append(spectrum_Eee)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    spectrums_all.insert(0, background_data)
    titles_all.insert(0, "^{8}B #nu_{e}")
    labels_all.insert(0, "^{8}B #nu_{e}")
    U = grid1d_U2M[0][0]

    plotSpectrums(
        spectrums_all,
        titles_all,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels_all,
        "energy,flux",
    )


def plot_nuL_El_costheta_decay_in_flight_grid1d(spectrum_L, grid1d_U2M, fileName):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 201
    costheta_step = 2.0 / (npoints_costheta - 1.0)
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * costheta_step

    labels_all = []
    titles_all = []

    diff_costheta_all = []
    diff_cosphi_all = []
    diff_El_all = []
    xtitle_El = "Neutrino #nu_{e} Energy (MeV)"
    xtitle_ct = "#nu_{e} angle cos(#theta_{Sun})"
    xtitle_cp = "#nu_{e} angle cos(#phi_{decay})"
    ytitle_El = "Neutrino Flux (MeV^{-1} cm^{-2} s^{-1})"
    ytitle_ct = "Neutrino Flux (cm^{-2} s^{-1})"

    xmin_El = 0.1
    xmax_El = 16.0
    ymin_El = 1e-4
    ymax_El = 1.0

    xmin_ct = -1.0
    xmax_ct = 1.0
    ymin_ct = 1e-4
    ymax_ct = 1.0

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append("m_{#nuH} = " + str(int(MH)) + " MeV")

        spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
        flux_R = spectrum_R[:, 1]
        index_max = np.argmax(flux_R)
        E_max_flux = energy[index_max]

        spectrum_L_left = np.zeros((len(energy), 2))

        diff_El_decayed = np.zeros((len(energy), 2))
        diff_El_decayed_inside = np.zeros((len(energy), 2))
        diff_El_decayed_outside = np.zeros((len(energy), 2))
        diff_costheta_decayed = np.zeros((npoints_costheta, 2))
        diff_costheta_decayed_inside = np.zeros((npoints_costheta, 2))
        diff_costheta_decayed_outside = np.zeros((npoints_costheta, 2))
        diff_cosphi_decayed = np.zeros((npoints_costheta, 2))
        diff_cosphi_decayed_inside = np.zeros((npoints_costheta, 2))
        diff_cosphi_decayed_outside = np.zeros((npoints_costheta, 2))
        for ie in range(len(energy)):
            spectrum_L_left[ie][0] = energy[ie]
            spectrum_L_left[ie][1] = spectrum_L[ie][1] - spectrum_R[ie][1]

            diff_El_decayed[ie][0] = energy[ie]
            diff_El_decayed_inside[ie][0] = energy[ie]
            diff_El_decayed_outside[ie][0] = energy[ie]
        for iTh in range(npoints_costheta):
            diff_costheta_decayed[iTh][0] = costheta_arr[iTh]
            diff_costheta_decayed_inside[iTh][0] = costheta_arr[iTh]
            diff_costheta_decayed_outside[iTh][0] = costheta_arr[iTh]
            diff_cosphi_decayed[iTh][0] = costheta_arr[iTh]
            diff_cosphi_decayed_inside[iTh][0] = costheta_arr[iTh]
            diff_cosphi_decayed_outside[iTh][0] = costheta_arr[iTh]

        print("===============================================")
        print("MH = ", MH, "U2 = ", U2)
        print("Tau_CM = ", RHN_TauCM(MH, U2))

        ratio_orbit = findRatioForDistance(MH, E_max_flux, U2, distance_SE)
        # split decay in steps
        # first, decay before reaching earth orbit
        nsteps_earth = 100
        distance_step = distance_SE * 1.0 / nsteps_earth
        for istep in range(nsteps_earth):
            (
                diff_El_this,
                diff_costheta_this,
                diff_cosphi_this,
                diff_El_costheta_this,
            ) = getNulEAndAngleFromRHNDecay(
                spectrum_R,
                MH,
                U2,
                istep * 1.0 * distance_step,
                distance_step,
                costheta_arr,
            )
            print("decay inside earth orbit, distance = ", '%.2f'%(istep*1.0*distance_step/distance_SE), " (SE), istep ", istep+1, "/", nsteps_earth, ", decayed flux = ", integrateSpectrum(diff_El_this))
            for ie in range(len(energy)):
                diff_El_decayed[ie][1] += diff_El_this[ie][1]
                diff_El_decayed_inside[ie][1] += diff_El_this[ie][1]
            for iTh in range(npoints_costheta):
                diff_costheta_decayed[iTh][1] += diff_costheta_this[iTh][1]
                diff_costheta_decayed_inside[iTh][1] += diff_costheta_this[iTh][1]
                diff_cosphi_decayed[iTh][1] += diff_cosphi_this[iTh][1]
                diff_cosphi_decayed_inside[iTh][1] += diff_cosphi_this[iTh][1]
            # print(diff_El_this)
            # print(diff_costheta_this)
            # print(diff_cosphi_this)
            # print("Total flux in this step: ", np.sum(diff_costheta_this[:,1]), np.sum(diff_cosphi_this[:,1]))

        # then, decay after flying outside earth orbit
        distance_start = distance_SE
        ratio_decayed = findRatioForDistanceSpectrum(MH, spectrum_R, U2, distance_start)
        distance_next = findDistanceForRatio(
            MH, E_max_flux, U2, ratio_orbit + (1 - ratio_orbit) / 10.0
        )

        print("Ratio decay inside earth orbit: ", ratio_decayed)

        while ratio_decayed < 0.999:
            (
                diff_El_this,
                diff_costheta_this,
                diff_cosphi_this,
                diff_El_costheta_this,
            ) = getNulEAndAngleFromRHNDecay(
                spectrum_R,
                MH,
                U2,
                distance_start,
                distance_next - distance_start,
                costheta_arr,
            )
            ratio_old = ratio_decayed
            ratio_decayed = findRatioForDistanceSpectrum(
                MH, spectrum_R, U2, distance_next
            )
            print("decay outside earth orbit, distance = ", '%.2f'%(distance_start/distance_SE), " (SE), fraction decayed: ", '%.3f' % ratio_decayed, ", decayed flux = ", integrateSpectrum(diff_El_this))

            distance_start = distance_next
            ratio_decayed2 = findRatioForDistance(MH, E_max_flux, U2, distance_next)
            ratio_remained2 = 1.0 - ratio_decayed2
            ratio_next = ratio_decayed2 + ratio_remained2 / 10.0
            if ratio_remained2 < 0.01:
                ratio_next = ratio_decayed2 + ratio_remained2 / 2.0
            distance_next = findDistanceForRatio(MH, E_max_flux, U2, ratio_next)

            for ie in range(len(energy)):
                diff_El_decayed[ie][1] += diff_El_this[ie][1]
                diff_El_decayed_outside[ie][1] += diff_El_this[ie][1]
            for iTh in range(npoints_costheta):
                diff_costheta_decayed[iTh][1] += diff_costheta_this[iTh][1]
                diff_costheta_decayed_outside[iTh][1] += diff_costheta_this[iTh][1]
                diff_cosphi_decayed[iTh][1] += diff_cosphi_this[iTh][1]
                diff_cosphi_decayed_outside[iTh][1] += diff_cosphi_this[iTh][1]
            # print(diff_El_this)
            # print(diff_costheta_this)
            # print(diff_cosphi_this)
            # print("Total flux in this step: ", np.sum(diff_costheta_this[:,1]), np.sum(diff_cosphi_this[:,1]))

        print("Total vH flux: ", integrateSpectrum(spectrum_R), np.sum(spectrum_R))
        print(
            "Flux of vH decay inside earth orbit (int El): ",
            integrateSpectrum(diff_El_decayed_inside),
            ", ",
            np.sum(diff_El_decayed_inside),
        )
        print(
            "Flux of vH decay inside earth orbit (int costheta): ",
            integrateSpectrum(diff_costheta_decayed_inside),
            ", ",
            np.sum(diff_costheta_decayed_inside),
        )
        print(
            "Flux of vH decay inside earth orbit (int cosphi): ",
            integrateSpectrum(diff_cosphi_decayed_inside),
            ", ",
            np.sum(diff_cosphi_decayed_inside),
        )
        print(
            "Flux of vH decay outside earth orbit (int El): ",
            integrateSpectrum(diff_El_decayed_outside),
            ", ",
            np.sum(diff_El_decayed_outside),
        )
        print(
            "Flux of vH decay outside earth orbit (int costheta): ",
            integrateSpectrum(diff_costheta_decayed_outside),
            ", ",
            np.sum(diff_costheta_decayed_outside),
        )
        print(
            "Flux of vH decay outside earth orbit (int cosphi): ",
            integrateSpectrum(diff_cosphi_decayed_outside),
            ", ",
            np.sum(diff_cosphi_decayed_outside),
        )

        # plot1DCurve(diff_costheta_decayed, "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV", xtitle_ct, ytitle_ct, fileName+"DecayInFlightNuLCosthetaSun_U"+str(U2)+"_M"+str(MH), xmin_ct, xmax_ct, ymin_ct, ymax_ct)
        # plot1DCurve(diff_costheta_decayed_inside, "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV", xtitle_ct, ytitle_ct, fileName+"DecayInFlightNuLCosthetaSun_U"+str(U2)+"_M"+str(MH)+"_Inside", xmin_ct, xmax_ct, ymin_ct, ymax_ct)
        # plot1DCurve(diff_costheta_decayed_outside, "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV", xtitle_ct, ytitle_ct, fileName+"DecayInFlightNuLCosthetaSun_U"+str(U2)+"_M"+str(MH)+"_Outside", xmin_ct, xmax_ct, ymin_ct, ymax_ct)
        # plotSpectrums([diff_costheta_decayed, diff_costheta_decayed_inside, diff_costheta_decayed_outside], ["Total", "#nu_{H} decay inside earth orbit #rightarrow #nu_{e} fly through detector", "#nu_{H} decay outside earth orbit #rightarrow #nu_{e} fly through detector", "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV"], xtitle_ct, ytitle_ct, fileName+"DecayInFlightNuLCosthetaSun_U"+str(U2)+"_M"+str(MH)+"_InsideOutside", xmin_ct, xmax_ct, ymin_ct, ymax_ct, labels=["Total", "Inside", "Outside"], column_names="costheta,flux", legNColumns=1)
        # plot1DCurve(diff_cosphi_decayed, "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV", xtitle_cp, ytitle_ct, fileName+"DecayInFlightNuLCosphiSun_U"+str(U2)+"_M"+str(MH), xmin_ct, xmax_ct, ymin_ct, ymax_ct)
        # plotSpectrums([diff_cosphi_decayed, diff_cosphi_decayed_inside, diff_cosphi_decayed_outside], ["Total", "#nu_{H} decay inside earth orbit #rightarrow #nu_{e} fly through detector", "#nu_{H} decay outside earth orbit #rightarrow #nu_{e} fly through detector", "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV"], xtitle_cp, ytitle_ct, fileName+"DecayInFlightNuLCosphiSun_U"+str(U2)+"_M"+str(MH)+"_InsideOutside", xmin_ct, xmax_ct, ymin_ct, ymax_ct, labels=["Total", "Inside", "Outside"], column_names="cosphi,flux", legNColumns=1)
        # plot1DCurve(diff_El_decayed, "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV", xtitle_El, ytitle_El, fileName+"DecayInFlightNuLEnergy_U"+str(U2)+"_M"+str(MH), xmin_El, xmax_El, ymin_El, ymax_El)
        # plotSpectrums([diff_El_decayed, diff_El_decayed_inside, diff_El_decayed_outside, spectrum_L, spectrum_L_left], ["Total (decay inside+outside)", "#nu_{H} decay inside earth orbit #rightarrow #nu_{e} fly through detector", "#nu_{H} decay outside earth orbit #rightarrow #nu_{e} fly through detector", "#nu_{e} from ^{8}B (original)", "#nu_{e} from ^{8}B (survived)", "U^{2} = "+str(U2)+", m_{#nuH} = "+str(int(MH))+" MeV"], xtitle_El, ytitle_El, fileName+"DecayInFlightNuLEnergy_U"+str(U2)+"_M"+str(MH)+"_InsideOutside", xmin_El, xmax_El, ymin_El, ymax_El, labels=["Total", "Inside", "Outside", "nuLFrom8B", "nuLFrom8BSurvived"], column_names="energy,flux", legNColumns=1)
        print("===============================================")
        print("Total flux of nuR generated:")
        print(np.sum(flux_R))
        print("===============================================")
        print("Total flux of nuL from nuR decay and reaches earth:")
        print(np.sum(diff_costheta_decayed[:, 1]))
        np.savetxt(
            f"./DecayData/DecayDetectorCountCostheta_U{U2}_M{MH}.csv",
            diff_costheta_decayed,
            delimiter=",",
        )
        print(np.sum(diff_cosphi_decayed[:, 1]))
        np.savetxt(
            f"./DecayData/DecayDetectorCountCosphi_U{U2}_M{MH}.csv",
            diff_cosphi_decayed[:, 1],
            delimiter=",",
        )
        print(np.sum(diff_El_decayed[:, 1]))
        np.savetxt(
            f"./DecayData/DecayDetectorCountEl_U{U2}_M{MH}.csv",
            diff_El_decayed,
            delimiter=",",
        )
        print("===============================================")

        diff_costheta_all.append(diff_costheta_decayed)
        diff_cosphi_all.append(diff_cosphi_decayed)
        diff_El_all.append(diff_El_decayed)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]
    titles_all.append("U^{2} = " + str(U))

    plotSpectrums(diff_costheta_all, titles_all, xtitle_ct, ytitle_ct, fileName+"DecayInFlightNuLCosthetaSun_U"+str(U)+"_AllMass", xmin_ct, xmax_ct, ymin_ct, ymax_ct, labels=labels_all, column_names="costheta,flux")
    plotSpectrums(diff_cosphi_all, titles_all, xtitle_cp, ytitle_ct, fileName+"DecayInFlightNuLCosphiSun_U"+str(U)+"_AllMass", xmin_ct, xmax_ct, ymin_ct, ymax_ct, labels=labels_all, column_names="cosphi,flux")

    diff_El_all.insert(0, spectrum_L)
    titles_all.insert(0, "#nu_{e} from ^{8}B")
    labels_all.insert(0, "nuLFrom8B")

    plotSpectrums(
        diff_El_all,
        titles_all,
        xtitle_El,
        ytitle_El,
        fileName + "DecayInFlightNuLEnergy_U" + str(U) + "_AllMass",
        xmin_El,
        xmax_El,
        ymin_El,
        ymax_El,
        labels=labels_all,
        column_names="energy,flux",
    )


def plot_nuL_costheta_grid1d_sampling(
    spectrum_L,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    num_samples=10000,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 203
    costheta_step = 0.01
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.01 + i * costheta_step
    costheta_arr[-2] = 1.0

    labels_all = []
    titles_all = []

    diff_costheta_all = []

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append(
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV"
        )

        spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
        flux_R = spectrum_R[:, 1]  # spectrum of nuH

        diff_El_this, diff_costheta_this, diff_Eee_this = getNuLEAndAngleBySampling(
            spectrum_R, MH, num_samples, costheta_arr
        )

        # normalize to 1
        sum_diff_costheta = np.sum(diff_costheta_this[:, 1])
        for icostheta in range(npoints_costheta):
            diff_costheta_this[icostheta][1] *= (
                (npoints_costheta - 1) * 1.0 / sum_diff_costheta
            )

        plot1DCurve(
            diff_costheta_this,
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV",
            xtitle,
            ytitle,
            fileName + "_U" + str(U2) + "_M" + str(MH),
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
        diff_costheta_all.append(diff_costheta_this)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]

    plotSpectrums(
        diff_costheta_all,
        titles_all,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels_all,
        "costheta,flux",
    )


# detector_S: cm^2
# detector_L: m
# exposure_time: s
def plot_Eee_in_detector_grid1d_sampling(
    spectrum_L,
    grid1d_U2M,
    xtitle,
    ytitle,
    fileName,
    detector_S,
    detector_L,
    exposure_time,
    num_samples=10000,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    canvas=None,
):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 203
    costheta_step = 0.01
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.01 + i * costheta_step
    costheta_arr[-2] = 1.0

    labels_all = []
    titles_all = []

    diff_Eee_all = []

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append("m_{#nuH} = " + str(int(MH)) + " MeV")

        spectrum_R_orig = getRHNSpectrum(spectrum_L, MH, U2)
        spectrum_R = getDecayedRHNSpectrum(
            spectrum_R_orig, MH, U2, distance_SE, detector_L
        )
        for ie in range(len(spectrum_R)):
            spectrum_R[ie][1] = spectrum_R[ie][1] * detector_S * exposure_time

        diff_El_this, diff_costheta_this, diff_Eee_this = getNuLEAndAngleBySampling(
            spectrum_R, MH, num_samples, costheta_arr
        )

        plot1DCurve(
            diff_Eee_this,
            "U^{2} = " + str(U2) + ", m_{#nuH} = " + str(int(MH)) + " MeV",
            xtitle,
            ytitle,
            fileName + "_U" + str(U2) + "_M" + str(MH),
            xmin,
            xmax,
            ymin,
            ymax,
            canvas,
        )
        diff_Eee_all.append(diff_Eee_this)
        print(MH, U2, np.sum(spectrum_R[:, 1]))
        # diff_Eee_all.append(spectrum_R)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]

    plotSpectrums(
        diff_Eee_all,
        titles_all,
        xtitle,
        ytitle,
        fileName + "_U" + str(U) + "_AllMass",
        xmin,
        xmax,
        ymin,
        ymax,
        canvas,
        labels_all,
        "energy,flux",
    )


def plot_nuL_El_costheta_grid1d_sampling(
    spectrum_L, grid1d_U2M, fileName, num_samples=10000
):
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 203
    costheta_step = 0.01
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.01 + i * costheta_step
    costheta_arr[-2] = 1.0

    labels_all = []
    titles_all = []

    diff_costheta_all = []
    diff_El_all = []
    xtitle_El = "Neutrino Energy (MeV)"
    xtitle_ct = "#nu_{e} emission angle cos(#theta)"
    ytitle_El = "Neutrino Flux (MeV^{-1} cm^{-2} s^{-1})"
    ytitle_ct = "r.u."

    xmin_El = 0.1
    xmax_El = 16.0
    ymin_El = 10.0
    ymax_El = 5.0e6

    xmin_ct = -1.0
    xmax_ct = 1.0
    ymin_ct = 1e-2
    ymax_ct = 40.0

    for ig in range(len(grid1d_U2M)):
        start_time = time.time()
        MH = grid1d_U2M[ig][1]
        U2 = grid1d_U2M[ig][0]

        labels_all.append("U" + str(U2) + "_M" + str(MH))
        titles_all.append(
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV"
        )

        spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)

        diff_El_this, diff_costheta_this, diff_Eee_this = getNuLEAndAngleBySampling(
            spectrum_R, MH, num_samples, costheta_arr
        )

        # normalize costheta to 1
        sum_diff_costheta = np.sum(diff_costheta_this[:, 1])
        for icostheta in range(npoints_costheta):
            diff_costheta_this[icostheta][1] *= (
                (npoints_costheta - 1) * 1.0 / sum_diff_costheta
            )

        plot1DCurve(
            diff_costheta_this,
            "U^{2} = " + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV",
            xtitle_ct,
            ytitle_ct,
            fileName + "nuLEmissionAngleSampling_U" + str(U2) + "_M" + str(MH),
            xmin_ct,
            xmax_ct,
            ymin_ct,
            ymax_ct,
        )

        diff_costheta_all.append(diff_costheta_this)

        spectrums_3types = []
        titles = [
            "#nu_{e} from ^{8}B",
            "#nu_{H} (U^{2}=" + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV)",
            "#nu_{e} from #nu_{H}",
        ]
        spectrums_3types.append(spectrum_L)
        spectrums_3types.append(spectrum_R)
        spectrums_3types.append(diff_El_this)

        titles_El3 = [
            "#nu_{e} from ^{8}B",
            "#nu_{H} (U^{2}=" + str(int(U2)) + ", m_{#nuH} = " + str(int(MH)) + " MeV)",
            "#nu_{e} from #nu_{H}",
        ]

        plotSpectrums(
            spectrums_3types,
            titles_El3,
            xtitle_El,
            ytitle_El,
            fileName + "nuLSpectrumSampling_U" + str(U2) + "_M" + str(MH),
            xmin_El,
            xmax_El,
            ymin_El,
            ymax_El,
        )
        diff_El_all.append(diff_El_this)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time used: ", elapsed_time, " seconds")

    U = grid1d_U2M[0][0]

    plotSpectrums(
        diff_costheta_all,
        titles_all,
        xtitle_ct,
        ytitle_ct,
        fileName + "nuLEmissionAngleSampling_U" + str(U) + "_AllMass",
        xmin_ct,
        xmax_ct,
        ymin_ct,
        ymax_ct,
        labels=labels_all,
        column_names="costheta,flux",
    )

    diff_El_all.insert(0, spectrum_L)
    titles_all.insert(0, "#nu_{e} from ^{8}B")
    plotSpectrums(
        diff_El_all,
        titles_all,
        xtitle_El,
        ytitle_El,
        fileName + "nuLSpectrumSampling_U" + str(U) + "_AllMass",
        xmin_El,
        xmax_El,
        ymin_El,
        ymax_El,
        labels=labels_all,
        column_names="energy,flux",
    )


def testSampling2D(spectrum, num_samples):
    print("sampling step 1")
    samples_generated, spectrum_generated = generate_samples_from_spectrum(
        spectrum, num_samples
    )
    energy = spectrum[:, 0]
    flux_R = spectrum[:, 1]
    npoints_costheta = 201
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * 2.0 / (npoints_costheta - 1)

    MH = 4.0
    Els = []
    costhetas = []
    Els_cms = []
    costhetas_cms = []

    maxDiff_lab = 0.0
    maxDiff_cms = 0.0
    for EH in energy:
        if EH > MH:
            maxDiff_lab_this = getMaximumValue2D(
                diff_El_costheta_lab, [0.0, 16.0], [1.00, -1.00], MH, EH
            )
            maxDiff_cms_this = getMaximumValue2D(
                diff_El_costheta_cms, [0.0, 16.0], [1.00, -1.00], MH, EH
            )
            if maxDiff_lab_this > maxDiff_lab:
                maxDiff_lab = maxDiff_lab_this
            if maxDiff_cms_this > maxDiff_cms:
                maxDiff_cms = maxDiff_cms_this

    print("maxDiff_lab: ", maxDiff_lab)
    print("maxDiff_cms: ", maxDiff_cms)

    print("sampling step 2")
    for EH in samples_generated:
        sample_this = rejection_sampling_2Dfunc(
            diff_El_costheta_lab,
            1,
            [0.0, 16.0],
            [-1.01, 1.01],
            maxDiff_lab * 2.0,
            MH,
            EH,
        )
        sample_this_cms = rejection_sampling_2Dfunc(
            diff_El_costheta_cms,
            1,
            [0.0, 16.0],
            [-1.01, 1.01],
            maxDiff_cms * 2.0,
            MH,
            EH,
        )
        El_this = sample_this[0][0]
        costheta_this = sample_this[0][1]

        El_this_cms, costheta_this_cms = cms_to_lab(
            sample_this_cms[0][0], sample_this_cms[0][1], MH, EH
        )
        Els.append(El_this)
        costhetas.append(costheta_this)
        Els_cms.append(El_this_cms)
        costhetas_cms.append(costheta_this_cms)

    print("sampling step 3")
    # based on integration
    diff_El_1 = np.zeros((len(energy), 2))
    for ieL in range(len(energy)):
        diff_El_1[ieL][0] = energy[ieL]
    for ieH in range(len(energy)):
        weight = flux_R[ieH]
        if weight < 1e-6:
            continue
        diff_ieL_temp = np.zeros(len(energy))
        for ieL in range(len(energy)):
            diff_ieL_temp[ieL] = diff_El(energy[ieL], MH, energy[ieH])
        sum_diff_ieL_temp = np.sum(diff_ieL_temp)
        for ieL in range(len(energy)):
            diff_El_1[ieL][1] += weight * diff_ieL_temp[ieL] / sum_diff_ieL_temp

    print("sampling step 4")
    # based on sampling
    diff_El_2 = np.zeros((len(energy), 2))
    for ieL in range(len(energy)):
        diff_El_2[ieL][0] = energy[ieL]
    energy_bin = np.append(energy, [100.0])
    Els_count, Els_edges = np.histogram(Els, bins=energy_bin)
    print("Els_count total: ", np.sum(Els_count), len(Els))
    sum_flux_R = np.sum(flux_R)
    sum_Els_count = np.sum(Els_count)
    for ieL in range(len(Els_count)):
        diff_El_2[ieL][1] = Els_count[ieL] * 1.0 * sum_flux_R / sum_Els_count

    print("sampling step 5")
    # based on sampling
    diff_El_3 = np.zeros((len(energy), 2))
    for ieL in range(len(energy)):
        diff_El_3[ieL][0] = energy[ieL]
    Els_count_cms, Els_edges_cms = np.histogram(Els_cms, bins=energy_bin)
    print("Els_count_cms total: ", np.sum(Els_count_cms), len(Els_cms))
    sum_flux_R = np.sum(flux_R)
    sum_Els_count_cms = np.sum(Els_count_cms)
    for ieL in range(len(Els_count_cms)):
        diff_El_3[ieL][1] = Els_count_cms[ieL] * 1.0 * sum_flux_R / sum_Els_count_cms

    plotSpectrums(
        [diff_El_1, diff_El_2, diff_El_3],
        ["integrated", "sampled in lab frame", "sampled in cms frame"],
        "Neutrino Energy (MeV)",
        "r.u.",
        "plots/testSampling2D_El",
        0.0,
        16.0,
        10.0,
        2.0e6,
    )


# The `testSampling2D` function performs a 2D sampling of neutrino energy and angle distributions in both the lab and center-of-mass (CMS) frames. Here's a concise breakdown of its steps:

# 1. **Initialization**:
#    - Generates samples from the input spectrum.
#    - Initializes arrays for energy (`energy`) and flux (`flux_R`).
#    - Sets up an array for cosine of the angle (`costheta_arr`).

# 2. **Maximum Difference Calculation**:
#    - For each energy above the neutrino mass (`MH = 4.0 MeV`), computes the maximum difference in the lab and CMS frames using `getMaximumValue2D`.

# 3. **Sampling**:
#    - Uses rejection sampling (`rejection_sampling_2Dfunc`) to generate samples in both lab and CMS frames.
#    - Transforms CMS samples to the lab frame using `cms_to_lab`.

# 4. **Integration-Based Distribution**:
#    - Computes the energy distribution based on integration (`diff_El_1`).

# 5. **Sampling-Based Distributions**:
#    - Computes energy distributions based on sampled data in the lab frame (`diff_El_2`) and CMS frame (`diff_El_3`).

# 6. **Plotting**:
#    - Plots the integrated and sampled distributions using `plotSpectrums`.

# This function is used to validate the sampling method by comparing integrated and sampled distributions. The results are saved in a plot file (`plots/testSampling2D_El`).
