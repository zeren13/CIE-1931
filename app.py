            if st.button("Configure", key=f"btn_cfg_{ds['id']}"):
                request_open_config(ds["id"])
        with c3:
            st.write(f"Wavelength: {cfg['wl_min']}-{cfg['wl_max']} nm | interp: {cfg['interp']} nm")

        try:
            df = ds["df"]
            if cfg["wl_min"] >= cfg["wl_max"]:
                raise ValueError("Minimum wavelength must be lower than maximum wavelength.")
            if cfg["wl_min"] < CMF_MIN or cfg["wl_max"] > CMF_MAX:
                raise ValueError(f"The range must be within the CMF domain: {CMF_MIN:.0f}-{CMF_MAX:.0f} nm.")
            if cfg["wl_col"] not in df.columns or cfg["int_col"] not in df.columns:
                raise ValueError("Selected columns do not exist in this file/sheet.")

            wl_grid, intensity_grid = preprocess_spectrum(
                df, cfg["wl_col"], cfg["int_col"],
                cfg["wl_min"], cfg["wl_max"], cfg["interp"],
                clip_negative=cfg["clip_negative"],
                baseline_subtract_min=cfg["baseline_subtract_min"],
                smooth_method=cfg["smooth_method"],
                smooth_window=cfg["smooth_window"],
                smooth_poly=cfg["smooth_poly"],
                normalize_mode=cfg["normalize"],
            )

            x_val, y_val = spectrum_to_xy(wl_grid, intensity_grid)
            wl_dom, purity, wl_kind = dominant_wavelength_and_purity(x_val, y_val, white_point=WHITE_POINT)

            ax.scatter([x_val], [y_val], c=[cfg["color"]], marker=cfg["marker"], s=cfg["size"], label=cfg["label"])
            if show_point_labels:
                ax.text(x_val + 0.008, y_val, cfg["label"],
                        fontsize=tick_font_size, fontfamily=tick_font_family,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))

            results.append({
                "Label": cfg["label"],
                "x": float(x_val),
                "y": float(y_val),
                "Wavelength_nm": (float(wl_dom) if np.isfinite(wl_dom) else np.nan),
                "Wavelength_type": wl_kind,
                "Excitation_purity_%": (float(purity) if np.isfinite(purity) else np.nan),
                "wl_min_nm": int(cfg["wl_min"]),
                "wl_max_nm": int(cfg["wl_max"]),
                "interp_nm": int(cfg["interp"]),
                "baseline_subtract_min": bool(cfg["baseline_subtract_min"]),
                "clip_negative": bool(cfg["clip_negative"]),
                "smooth_method": cfg["smooth_method"],
                "smooth_window": int(cfg["smooth_window"]),
                "smooth_poly": int(cfg["smooth_poly"]),
                "normalize": cfg["normalize"],
                "wl_column": cfg["wl_col"],
                "intensity_column": cfg["int_col"],
            })

            spectra_to_plot.append({
                "label": cfg["label"],
                "wl": wl_grid,
                "it": intensity_grid,
                "color": cfg.get("spectrum_color", cfg["color"])
            })

            st.success(f"{cfg['label']}: x={x_val:.4f}, y={y_val:.4f} | wavelength ({wl_kind})={wl_dom:.1f} nm | purity={purity:.1f}%")
        except Exception as e:
            st.error(f"{cfg['label']}: {e}")
else:
    st.info("Upload files to begin.")

# ----------------- Outputs -----------------
if results:
    try:
        legend = ax.legend(loc=legend_loc, fontsize=legend_font_size, ncol=legend_ncols)
        frame = legend.get_frame()
        frame.set_facecolor(legend_box_color)
        frame.set_alpha(legend_box_alpha)
        frame.set_linewidth(legend_box_linewidth)
    except Exception:
        pass

    st.markdown("### Analysis overview")
    results_df = pd.DataFrame(results)
    minimal_cols = [
        "Label",
        "x",
        "y",
        "Wavelength_nm",
        "Wavelength_type",
        "Excitation_purity_%",
    ]
    minimal_cols = [c for c in minimal_cols if c in results_df.columns]
    table_df = results_df[minimal_cols].copy()
    for c in ["x", "y"]:
        if c in table_df.columns:
            table_df[c] = table_df[c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    for c in ["Wavelength_nm", "Excitation_purity_%"]:
        if c in table_df.columns:
            table_df[c] = table_df[c].map(lambda v: f"{v:.1f}" if np.isfinite(v) else "")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("### CIE 1931 diagram")
        try:
            plt.tight_layout()
        except Exception:
            pass
        st.pyplot(fig)

    with right_col:
        st.markdown("### Emission spectra")
        if spectra_to_plot:
            fig_s, ax_s = plt.subplots(figsize=(5.5, 3.2))
            for s in spectra_to_plot:
                ax_s.plot(s["wl"], s["it"], label=s["label"], color=s["color"], linewidth=1.8)
            ax_s.set_xlabel("Wavelength (nm)")
            ax_s.set_ylabel("Intensity (a.u.)")
            ax_s.tick_params(labelsize=9)
            ax_s.grid(alpha=0.25)
            try:
                ax_s.legend(fontsize=7, loc="best")
            except Exception:
                pass
            st.pyplot(fig_s)
        else:
            st.info("No spectra to display.")

        st.markdown("### Coordinate table")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        with st.expander("Downloads", expanded=False):
            csv_buf = io.StringIO()
            results_df.to_csv(csv_buf, index=False)
            st.download_button("Download full CSV table", data=csv_buf.getvalue(), file_name="CIE1931_coordinates.csv", mime="text/csv")

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="tiff", dpi=dpi_save)
            img_buf.seek(0)
            st.download_button("Download TIFF diagram", data=img_buf.getvalue(), file_name="CIE1931_diagram.tiff", mime="image/tiff")
