import os
import numpy as np
import pandas as pd
from pytools.rt_ploter import rt


def _find_u2_at_count_1(mh_vals, u2_vals, count_grid):
    """Find U2 where signal count = 1 for each MH.

    Returns lower (up-crossing) and upper (down-crossing) boundaries.
    Some MH may have none, one, or two crossings.

    Parameters
    ----------
    mh_vals : ndarray (nMH,)
    u2_vals : ndarray (nU2,)
    count_grid : ndarray (nMH, nU2)

    Returns
    -------
    mh_low : list
        MH for lower boundary (up-crossing)
    u2_low : list
        Corresponding U2 values
    mh_high : list
        MH for upper boundary (down-crossing)
    u2_high : list
        Corresponding U2 values
    """
    count_safe = np.maximum(count_grid, 1e-300)
    log_u2 = np.log10(u2_vals)
    log_count = np.log10(count_safe)

    mh_low, u2_low = [], []
    mh_high, u2_high = [], []

    for imh in range(len(mh_vals)):
        row = log_count[imh]
        above = row >= 0.0  # count >= 1

        if not np.any(above):
            # No crossing at all
            continue

        # Find all transitions
        transitions = np.where(np.diff(above.astype(int)) != 0)[0] + 1

        for i_t in transitions:
            u_lo = log_u2[i_t - 1]
            u_hi = log_u2[i_t]
            c_lo = row[i_t - 1]
            c_hi = row[i_t]
            log_u_cross = np.interp(0.0, [c_lo, c_hi], [u_lo, u_hi])
            u_cross = 10.0**log_u_cross

            if above[i_t]:  # up-crossing (< 1 → > 1)
                mh_low.append(float(mh_vals[imh]))
                u2_low.append(u_cross)
            else:  # down-crossing (> 1 → < 1)
                mh_high.append(float(mh_vals[imh]))
                u2_high.append(u_cross)

    return mh_low, u2_low, mh_high, u2_high


def main():
    # ======================================================================
    # Read eepair signal count CSV and compute count=1 contours
    # ======================================================================
    # eepair_csv = "data/eepair_signal_count_cut.csv"
    eepair_csv = "data/s1_eepair_signal_count_cut.csv"
    print(f">>> Reading eepair CSV: {eepair_csv}")
    df_ee = pd.read_csv(eepair_csv)
    mh_ee = df_ee["mass_mev"].values

    u2_cols = [c for c in df_ee.columns if c.startswith("U2=")]
    u2_ee = np.array([float(c.split("=")[1]) for c in u2_cols])
    count_grid = df_ee[u2_cols].values

    mh_low, u2_low, mh_high, u2_high = _find_u2_at_count_1(
        mh_ee, u2_ee, count_grid
    )
    print(f"    Lower boundary (up-crossing):  {len(mh_low)} points")
    print(f"    Upper boundary (down-crossing): {len(mh_high)} points")
    print()

    # Save both contours
    sens_csv_low = "data/eepair_sensitivity_contour_low.csv"
    np.savetxt(
        sens_csv_low,
        np.column_stack([mh_low, u2_low]),
        delimiter=",",
        header="mH,u2_sens_low",
        comments="",
    )
    print(f">>> Saved lower sensitivity contour: {sens_csv_low}")

    if len(mh_high) > 0:
        sens_csv_high = "data/eepair_sensitivity_contour_high.csv"
        np.savetxt(
            sens_csv_high,
            np.column_stack([mh_high, u2_high]),
            delimiter=",",
            header="mH,u2_sens_high",
            comments="",
        )
        print(f">>> Saved upper sensitivity contour: {sens_csv_high}")
    print()

    # ======================================================================
    # ROOT plot
    # ======================================================================

    # ---- Exclusion band ----
    # excl_csv = "plots/upper_limit_new/upper_limit_bands_lushan.csv"
    # excl_csv = "plots/upper_limit_new/upper_limit_bands_combined.csv"
    excl_csv = "plots/upper_limit_new/upper_limit_bands.csv"
    print(f">>> Reading exclusion CSV: {excl_csv}")
    data = np.loadtxt(excl_csv, delimiter=",", skiprows=1)
    mh_excl = data[:, 0]
    u2_low_excl = data[:, 1]
    u2_high_excl = data[:, 2]

    y_max = np.max(u2_low_excl) * 0.8
    valid = u2_low_excl < y_max
    window = valid & (u2_high_excl < y_max)

    # ---- Canvas ----
    c = rt.TCanvas("c_excl", "Exclusion + eepair sensitivity", 1600, 1200)
    c.SetRightMargin(0.06)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.08)
    c.SetLogy(1)

    y_min = 5e-7
    y_frame = 1e0
    frame = c.DrawFrame(2.0, y_min, 15.0, y_frame) # plot interval
    frame.GetXaxis().SetTitle("m_{H}  [MeV]")
    frame.GetYaxis().SetTitle("|U_{eH}|^{2}")
    frame.GetYaxis().SetTitleOffset(1.2)
    frame.SetTitle("")

    # ---- Exclusion lower boundary ----
    n_low = int(np.sum(valid))
    gr_low = rt.TGraph(n_low)
    j = 0
    for i in range(len(mh_excl)):
        if valid[i]:
            gr_low.SetPoint(j, float(mh_excl[i]), float(u2_low_excl[i]))
            j += 1
    gr_low.SetLineColor(rt.kAzure + 2)
    gr_low.SetLineWidth(3)

    # ---- Exclusion upper boundary (window) ----
    n_win = int(np.sum(window))
    gr_high_excl = None
    if n_win > 0:
        gr_high_excl = rt.TGraph(n_win)
        j = 0
        for i in range(len(mh_excl)):
            if window[i]:
                gr_high_excl.SetPoint(j, float(mh_excl[i]), float(u2_high_excl[i]))
                j += 1
        gr_high_excl.SetLineColor(rt.kAzure + 2)
        gr_high_excl.SetLineWidth(3)

    y_floor = y_min * 0.5

    # ---- Exclusion lower boundary line ----
    x_valid = [float(mh_excl[i]) for i in range(len(mh_excl)) if valid[i]]
    y_low_plot = [max(y_floor, float(u2_low_excl[i])) for i in range(len(mh_excl)) if valid[i]]
    if n_low > 0:
        gl = rt.TGraph(len(x_valid))
        for i in range(len(x_valid)):
            gl.SetPoint(i, x_valid[i], y_low_plot[i])
        gl.SetLineColor(rt.kAzure + 2)
        gl.SetLineWidth(3)
        gl.SetMarkerSize(0)
        gl.Draw("L same")

    # ---- Exclusion upper boundary line ----
    x_win = [float(mh_excl[i]) for i in range(len(mh_excl)) if window[i]]
    y_win_plot = [max(y_floor, min(y_frame, float(u2_high_excl[i]))) for i in range(len(mh_excl)) if window[i]]
    if len(x_win) > 0:
        gh = rt.TGraph(len(x_win))
        for i in range(len(x_win)):
            gh.SetPoint(i, x_win[i], y_win_plot[i])
        gh.SetLineColor(rt.kAzure + 2)
        gh.SetLineWidth(3)
        gh.SetMarkerSize(0)
        gh.Draw("L same")

    # ---- Right closure ----
    if n_low > 0:
        y_right_high = y_win_plot[-1] if len(x_win) > 0 else y_frame
        gr_right = rt.TGraph(2)
        gr_right.SetPoint(0, x_valid[-1], y_low_plot[-1])
        gr_right.SetPoint(1, x_valid[-1], y_right_high)
        gr_right.SetLineColor(rt.kAzure + 2)
        gr_right.SetLineWidth(3)
        gr_right.SetMarkerSize(0)
        gr_right.Draw("L same")

    # ---- Borexino published ----
    borexino_csv = "./data/Borexino_exclusion.csv"
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

    # ---- eepair count=1 lower boundary ----
    gr_sens = None
    if len(mh_low) > 0:
        gr_sens = rt.TGraph(len(mh_low))
        for i in range(len(mh_low)):
            gr_sens.SetPoint(i, mh_low[i], u2_low[i])
        gr_sens.SetLineColor(rt.kCyan + 2)
        gr_sens.SetLineWidth(3)
        gr_sens.SetLineStyle(1)
        gr_sens.SetMarkerSize(0)
        gr_sens.Draw("L same")

    # ---- eepair count=1 upper boundary ----
    if len(mh_high) > 0:
        gr_sens_high = rt.TGraph(len(mh_high))
        for i in range(len(mh_high)):
            gr_sens_high.SetPoint(i, mh_high[i], u2_high[i])
        gr_sens_high.SetLineColor(rt.kCyan + 2)
        gr_sens_high.SetLineWidth(3)
        gr_sens_high.SetLineStyle(1)
        gr_sens_high.SetMarkerSize(0)
        gr_sens_high.Draw("L same")

    # ---- Close the red contour on the right side (like the blue band) ----
    if len(mh_low) > 0 and len(mh_high) > 0:
        # Find the rightmost MH common to both boundaries
        common_mh = sorted(set(mh_low) & set(mh_high))
        if common_mh:
            mh_right = common_mh[-1]
            # Get U2 at this MH from both boundaries
            idx_low = mh_low.index(mh_right)
            idx_high = mh_high.index(mh_right)
            u2_right_low = u2_low[idx_low]
            u2_right_high = u2_high[idx_high]

            gr_close = rt.TGraph(2)
            gr_close.SetPoint(0, mh_right, u2_right_low)
            gr_close.SetPoint(1, mh_right, u2_right_high)
            gr_close.SetLineColor(rt.kCyan + 2)
            gr_close.SetLineWidth(3)
            gr_close.SetLineStyle(1)
            gr_close.SetMarkerSize(0)
            gr_close.Draw("L same")

    # ---- S2 exclusion (from scattered neutrinos) ----
    s2_csv = "plots_grid_scan_s2_u2_41_mh_41/s2_upper_limit/s2_expected_exclusion.csv"
    gr_s2 = None
    if os.path.exists(s2_csv):
        s2_data = np.loadtxt(s2_csv, delimiter=",", skiprows=1)
        s2_mh = s2_data[:, 0]
        s2_u2 = s2_data[:, 1]
        gr_s2 = rt.TGraph(len(s2_mh))
        for i in range(len(s2_mh)):
            gr_s2.SetPoint(i, float(s2_mh[i]), float(s2_u2[i]))
        gr_s2.SetLineColor(rt.kOrange + 7)
        gr_s2.SetLineWidth(3)
        gr_s2.SetLineStyle(1)
        gr_s2.SetMarkerSize(0)
        gr_s2.Draw("L same")
    else:
        print(f"  S2 exclusion file not found: {s2_csv}")

    # ---- Legend ----
    leg = rt.TLegend(0.62, 0.72, 0.90, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    # leg.SetNColumns(2)
    if gr_ref:
        leg.AddEntry(gr_ref, "Borexino (published)", "l")
    leg.AddEntry(gr_low, "500t 1yr (e^{+}e^{-} energy)", "l")
    if gr_sens:
        leg.AddEntry(gr_sens, "500t 1yr (cos#theta_{e^{+}e^{-}} cut)", "l")
    if gr_s2:
        leg.AddEntry(gr_s2, "500t 1yr (decayed #nu)", "l")
    leg.Draw()

    # ---- Save ----
    outdir = "plots/upper_limit_new/"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "upper_limit_with_eepair_e.pdf")
    c.SaveAs(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
