"""
Tools for the analysis of RF pulse waveforms and prediction of
corresponding magnetization profiles.
"""
__all__ = ["pulse_analysis"]

import math
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate

from ._utils import gamma


def pulse_analysis(
    rf_envelope: npt.NDArray,
    duration: float,
    flip_angle: float = 1.0,
    npts: int = None,
    verbose: bool = False,
):
    """
    Analyze RF pulse and retrieve pulse power spectrum,
    number of bands, frequency offset and power deposition for each band,
    main lobe slice profile (small-angle approx), total pulse duration
    and isodelay.

    Args:
        rf_envelope: time envelope of RF pulse of shape (nchannels, npts).
        duration: RF duration in [ms].
        flip_angle: desired flip angle in [deg].
        npts: number of points for main lobe slice profile (default: npts).
        verbose: if True, display retrieved info.

    Returns:
        info: dict with the following fields:
            - 'duration': pulse duration in [s].
            - 'isodelay': pulse isodelay in [s].
            - 'freq_offset': frequency offset for each band in [Hz].
            - 'b1rms': root-mean-squared B1 for each band in [T]
        rf_main_lobe_profile: main lobe frequency profile (normalized to profile[f=0] = 1),
                              interpolated to npts.
        f_main_lobe: main lobe frequency axis in [Hz] (interpolated to npts).
        rf_full_spectrum_profile: full RF spectrum (normalized to profile[f=0] = 1).
        f_full_spectrum: full spectrum frequency axis in [Hz].
    """
    try:
        rf_envelope = rf_envelope.clone().numpy()
    except:
        pass

    # sum over channels for PTX pulses
    if len(rf_envelope.shape) == 2:
        rf_envelope = abs(rf_envelope.sum(axis=0))

    # get dt
    dt = float(duration) / len(rf_envelope) * 1e-3  # ms -> s

    # preserve input and normalize
    rf_envelope = rf_envelope.copy() / rf_envelope.sum()

    # get pulse number of points
    # duration = PulseFourierAnalysis.calc_pulse_duration(rf_envelope, dt)
    npts_original = len(rf_envelope)

    # get pulse isodelay
    isodelay = PulseFourierAnalysis.calc_isodelay(rf_envelope, dt)

    # get filtered pulse spectrum
    filt_frequency_profile, f_filt = PulseFourierAnalysis.calc_rf_frequency_profile(
        rf_envelope, dt, 10 * npts_original, filt=True
    )

    # get main lobe spectrum
    _, main_lobe = PulseFourierAnalysis.calc_bandwidth(
        np.abs(filt_frequency_profile), f_filt
    )

    # get number of bands and their separation
    freq_offset, nbands = PulseFourierAnalysis.calc_bands_frequency_offsets(
        np.abs(filt_frequency_profile), f_filt, main_lobe
    )

    # get raw pulse spectrum
    raw_frequency_profile, f_raw = PulseFourierAnalysis.calc_rf_frequency_profile(
        rf_envelope, dt, 10 * npts_original
    )

    # get pulse frequency width
    main_lobe_bw, _ = PulseFourierAnalysis.calc_bandwidth(
        np.abs(raw_frequency_profile), f_raw
    )

    # total bandwidth
    if nbands != 1:
        delta = np.diff(freq_offset).mean()
    else:
        delta = 0.0

    if nbands % 2 == 0:
        bw = delta * (nbands + 1) + 2 * main_lobe_bw
    else:
        bw = delta * nbands + 2 * main_lobe_bw

    # get main lobe envelope
    rf_envelope_main_lobe = PulseFourierAnalysis.calc_main_lobe_envelope(
        raw_frequency_profile, f_raw, main_lobe_bw
    )

    # calculate pulse scaling according to profile
    scale = PulseFourierAnalysis.calc_pulse_scaling(
        rf_envelope_main_lobe, dt, flip_angle
    )  # [T / a.u.]

    # calculate maximum B1
    b1_max = PulseFourierAnalysis.calc_b1max(scale * rf_envelope)  # [T]

    # calculate total power deposition
    b1_sqrd_tau_total = PulseFourierAnalysis.calc_b1sqr(
        scale * rf_envelope, dt
    )  # [T**2 * s]

    # calculate on-resonance power deposition
    b1_sqrd_tau_on = PulseFourierAnalysis.calc_b1sqr(
        scale * rf_envelope_main_lobe, dt
    )  # [T**2 * s]

    # assign power deposition to each band
    if nbands > 1:
        b1_sqrd_tau_off = (b1_sqrd_tau_total - b1_sqrd_tau_on) / (nbands - 1)
        beta = (b1_sqrd_tau_total - b1_sqrd_tau_on) / b1_sqrd_tau_on
        power_deposition = b1_sqrd_tau_off * np.ones(freq_offset.shape)
        power_deposition[freq_offset == 0] = b1_sqrd_tau_on
    else:
        power_deposition = b1_sqrd_tau_total  # [T**2 * s]

    # calculate b1rms in [uT]
    b1rms = 1e6 * (power_deposition / (duration * 1e-3)) ** 0.5

    # calculate number of points to have nbands * len(rf_pulse) points inside bandwidth
    if nbands % 2 == 0:
        nranges = nbands + 1
    else:
        nranges = nbands

    npts_interp = PulseFourierAnalysis.zoom_region(dt, nranges * npts_original, bw)
    rf_frequency_profile, f = PulseFourierAnalysis.calc_rf_frequency_profile(
        rf_envelope, dt, npts_interp
    )

    # get full power spectrum
    rf_full_spectrum_profile, f_full_spectrum = PulseFourierAnalysis.crop_region(
        rf_frequency_profile, f, nranges * npts_original
    )

    # get main lobe profile
    rf_main_lobe_profile, f_main_lobe = PulseFourierAnalysis.crop_region(
        rf_frequency_profile, f, npts_original
    )

    # resample main lobe profile
    rf_main_lobe_profile, f_main_lobe = PulseFourierAnalysis.resample_rf_profile(
        rf_main_lobe_profile, f_main_lobe, npts
    )

    # pack output
    info = {
        "dt": dt,
        "isodelay": isodelay,
        "freq_offset": freq_offset,
        "b1rms": b1rms,
        "main_lobe_bandwidth": main_lobe_bw,
        "spectrum_width": bw,
        "b1_max": b1_max,
        "flip_angle": flip_angle,
    }

    # print info
    if verbose is True:
        print(f"RF pulse duration: {round(duration*1e3, 2)} [ms]")
        print(f"RF pulse isodelay: {round(isodelay*1e3, 2)} [ms]")
        print(f"Main RF lobe bandwidth: {round(float(main_lobe_bw)*1e-3, 2)} [kHz]")
        print(f"Time-Bandwidth product: {round(float(duration * main_lobe_bw), 2)}")
        print(f"Number of bands in RF pulse: {nbands}")

        if nbands > 1:
            freq_off_kHz = (1e-3 * freq_offset).tolist()
            freq_off_kHz = [round(freq, 2) for freq in freq_off_kHz]
            b1rms_tot = float((b1rms**2).sum() ** 0.5)
            b1rms = b1rms.tolist()
            b1rms = [round(pw, 2) for pw in b1rms]
            b1rms_tot = round(b1rms_tot, 2)
            delta = round(float(delta) * 1e-3, 2)
            main_bw = round(float(main_lobe_bw) * 1e-3, 2)

            # actual print
            print(f"Bands frequency offsets: {freq_off_kHz} [kHz]")
            print(f"Bands frequency separation: {delta} [kHz]")
            print(f"Slice separation: {round(delta / main_bw, 2)} [# slices]")
            print(f"Bands B1 rms: {b1rms} [uT]")
            print(
                f"Ratio between off-resonance and on-resonance energy: {round(beta, 2)}"
            )
            print(f"B1 rms: {b1rms_tot} [uT]")

        else:
            print(f"RF pulse frequency offsets: {round(freq_offset*1e-3, 2)} [kHz]")
            print(f"B1 rms: {round(1e6*(power_deposition / duration)**0.5, 2)} [uT]")

        print(f"B1 max: {round(1e6 * b1_max, 2)} [uT]")

    return (
        info,
        rf_main_lobe_profile,
        f_main_lobe,
        rf_full_spectrum_profile,
        f_full_spectrum,
    )


class PulseFourierAnalysis:
    """
    Utils to analyze RF pulse and retrieve pulse power spectrum,
    number of bands, frequency offset and power deposition for each band,
    main lobe slice profile (small-angle approx), total pulse duration
    and isodelay.
    """

    @staticmethod
    def calc_pulse_duration(rf_envelope: np.ndarray, dt: float) -> float:
        """
        Calculate RF pulse duration.

        Args:
            rf_envelope: time envelope of RF pulse.
            dt: RF dwell time in [s].

        Returns:
            duration of RF pulse in [s].
        """
        n = len(rf_envelope)
        return n * dt

    @staticmethod
    def calc_isodelay(rf_envelope: np.ndarray, dt: float) -> Tuple[float, float]:
        """
        Calculate the time point of the effective rotation defined as the peak of the radio-frequency amplitude for the
        shaped pulses and the center of the pulse for the block pulses. Zero padding in the radio-frequency pulse is
        considered as a part of the shape. Delay field of the radio-frequency object is not taken into account.

        Args:
            rf_envelope: time envelope of RF pulse in [T].
            dt: waveform dwell time in [s].

        Returns:
            isodelay : Time between peak and end of the radio-frequency pulse in [s].
        """
        # preserve input
        rf_envelope = rf_envelope.copy()

        # get time
        t = dt * np.arange(len(rf_envelope))

        # We detect the excitation peak; if i is a plateau we take its center
        rf_max = max(abs(rf_envelope))
        i_peak = np.where(abs(rf_envelope) >= rf_max * 0.99999)[0]

        # get isodelay and corresponding index
        isodelay_idx = i_peak[len(i_peak) // 2]
        isodelay = t[-1] - t[isodelay_idx]

        return isodelay

    @staticmethod
    def calc_rf_frequency_profile(
        rf_envelope: np.ndarray, dt: float, npts: int = None, filt: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate n-points RF profile in frequency domain.

        Args:
            rf_envelope: time envelope of RF pulse.
            dt: RF dwell time in [s].
            npts: number of points in frequency domain (optional; default: len(rf_profile))

        Returns:
            rf_frequency_profile: n-points RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).
        """
        # preserve input
        rf_envelope = rf_envelope.copy()

        # get number of points
        if npts is None:
            npts = len(rf_envelope)

        # get frequency axis
        frange = 1 / dt
        f = np.linspace(-frange / 2, frange / 2, npts)

        # pad pulse to fit inside the desired range
        if npts > len(rf_envelope):
            sz = int(math.ceil((npts - len(rf_envelope)) / 2))

            if filt is True:
                window = np.hanning(len(rf_envelope))
                rf_envelope = window * rf_envelope

            rf_envelope = np.concatenate(
                (
                    np.zeros(sz, dtype=rf_envelope.dtype),
                    rf_envelope,
                    np.zeros(sz, dtype=rf_envelope.dtype),
                )
            )[:npts]

        # perform centered fourier transform
        rf_frequency_profile = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rf_envelope)))
        rf_frequency_profile = (
            rf_frequency_profile / rf_frequency_profile[len(rf_frequency_profile) // 2]
        )

        return rf_frequency_profile, f

    @staticmethod
    def calc_bands_frequency_offsets(
        rf_frequency_profile: np.ndarray, f: np.ndarray, main_lobe_profile: np.ndarray
    ) -> Union[np.ndarray, float]:
        """
        Calculate number of frequency bands and the corresponding frequency offset.

        Args:
            rf_frequency_profile: n-points RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).

        Returns:
            frequency offset for each band in [Hz].
        """
        # compute correlation
        corr = correlate(rf_frequency_profile, main_lobe_profile, "same")
        corr = corr / corr[len(corr) // 2]

        # get peaks
        peaks = np.isclose(corr, corr.max())

        # set also middle peak
        peaks[len(peaks) // 2] = True

        # count bands
        nbands = peaks.sum()
        nbands = int(nbands)

        # estimate distance between bands
        if nbands > 1:
            i_peaks = np.where(peaks)
            delta = float(np.diff(f[i_peaks]).mean())
        else:
            delta = 0.0

        # build frequency offset
        if nbands > 1:
            freq_offset = delta * np.arange(-nbands // 2 + 1, nbands // 2 + 1)
        else:
            freq_offset = 0.0

        # even bands case
        if nbands % 2 == 0:
            min_left = np.abs(f[i_peaks][: nbands // 2]).min()
            min_right = np.abs(f[i_peaks][nbands // 2 :]).min()
            if min_right < min_left:
                freq_offset = np.flip(-freq_offset)

        return freq_offset, nbands

    @staticmethod
    def zoom_region(dt: float, npts: int, bw: float) -> int:
        """
        Calculate number of points required to have nbands * len(pulse_envelope) points within desired frequency range.

        Args:
            dt: RF dwell time in [s].
            npts: number of points within the desired range.
            bw: desired frequency range in [Hz].

        Returns:
            total number of points to have npts points within desired frequency range.
        """
        # get new frequency step
        df = bw / npts

        return int((1 / df) / dt)

    @staticmethod
    def crop_region(
        rf_frequency_profile: np.ndarray, f: np.ndarray, npts: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop central region of a spectrum and its frequency axis.

        Args:
            rf_frequency_profile: RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).
            npts: number of desired points within the central region.

        Returns:
            rf_frequency_profile: n-points RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: cropped frequency axis in [Hz] (-Fcrop : Fcrop).
        """
        # preserve inputs
        rf_frequency_profile = rf_frequency_profile.copy()
        f = f.copy()

        # get number of points
        if npts is None:
            npts = len(rf_frequency_profile)

        if npts < len(rf_frequency_profile):
            # get indexes of desired region
            fcenter = int(len(rf_frequency_profile) // 2)
            fstart = int(fcenter - npts / 2)
            fstop = int(fstart + npts)

            # crop
            rf_frequency_profile = rf_frequency_profile[fstart:fstop]
            f = f[fstart:fstop]

        # normalize
        fcenter = int(len(rf_frequency_profile) // 2)
        rf_frequency_profile = rf_frequency_profile / rf_frequency_profile[fcenter]

        return rf_frequency_profile, f

    @staticmethod
    def resample_rf_profile(
        rf_frequency_profile: np.ndarray, f: np.ndarray, npts: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample profile to desired number of points.

        Args:
            rf_frequency_profile: RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).
            npts: number of desired points within the central region.

        Returns:
            rf_frequency_profile: n-points RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: resampled frequency axis in [Hz] (-Fcrop : Fcrop).
        """
        # preserve inputs
        rf_frequency_profile = rf_frequency_profile.copy()
        f = f.copy()

        if npts is not None and npts != len(rf_frequency_profile):
            fp = np.linspace(f[0], f[-1], npts)
            rf_frequency_profile = np.interp(fp, f, rf_frequency_profile)
            f = fp

        return rf_frequency_profile, f

    @staticmethod
    def calc_bandwidth(rf_frequency_profile: np.ndarray, f: np.ndarray) -> float:
        """
        Calculate full power spectrum width and main lobe width.

        Args:
            rf_frequency_profile: RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).

        Returns:
            main_lobe_bw: main lobe frequency width in [Hz].
        """
        # calc center
        center = len(rf_frequency_profile) // 2

        # separate positive and negative spectrum
        pos_spectrum = rf_frequency_profile[center:]
        neg_spectrum = np.flip(rf_frequency_profile[:center])

        # get minimum and maximum index for BW
        i_min = center - 1 - np.where(neg_spectrum <= 0.5)[0][0]
        i_max = center + np.where(pos_spectrum <= 0.5)[0][0]

        # estimate main lobe bandwidth
        main_lobe_bw = f[i_max] - f[i_min]

        # get minimum and maximum index for main lobe region
        npts = len(f[i_min:i_max])
        center = int(len(f) // 2)
        i_min = int(center - npts // 2)
        i_max = int(center + npts // 2)

        # get main lobe
        main_lobe = rf_frequency_profile[i_min:i_max]

        return main_lobe_bw, main_lobe

    @staticmethod
    def calc_main_lobe_envelope(
        rf_frequency_profile: np.ndarray, f: np.ndarray, main_lobe_bw: float
    ) -> np.ndarray:
        """
        Calculate main lobe time envelope

        Args:
            rf_frequency_profile: RF profile in frequency domain,
                                  (normalized so that rf_frequency_profile[f=0] = 1).
            f: frequency axis in [Hz] (-Fmax : Fmax; Fmax = 1/dt).
            main_lobe_bw: main lobe frequency width in [Hz].

        Returns:
            main lobe time envelope.
        """
        # get box function for main lobe
        main_lobe_freq = np.abs(f) <= main_lobe_bw

        # get main lobe envelope
        return np.fft.ifftshift(
            np.fft.ifft(np.fft.ifftshift(rf_frequency_profile * main_lobe_freq))
        )

    @staticmethod
    def calc_pulse_scaling(
        rf_envelope: np.ndarray, dt: float, flip_angle: float
    ) -> float:
        """
        Calculate RF scaling to obtain desired flip angle.

        Args:
            rf_envelope: time envelope of RF pulse.
            dt: RF dwell time in [s].
            flip_angle: desired flip angle in [deg].

        Returns:
            RF scaling in T / [a.u.] (i.e. converts arbitrary pulse unit to T)
        """
        # calculate current pulse area
        pulse_area = rf_envelope.sum()

        return np.deg2rad(flip_angle) / (gamma * dt) / pulse_area / 1e6

    @staticmethod
    def calc_b1sqr(rf_envelope: np.ndarray, dt: float) -> float:
        """
        Calculate integral B1+ squared  in [T**2 * s].

        Args:
            rf_envelope: time envelope of RF pulse in [T].
            dt: RF dwell time in [s].

        Returns:
            integral B1 squared in [T**2 * s].
        """
        return float((np.abs(rf_envelope) ** 2).sum()) * dt

    @staticmethod
    def calc_b1max(rf_envelope: np.ndarray) -> float:
        """
        Calculate maximum B1+  in [T].

        Args:
            rf_envelope: time envelope of RF pulse in [T].

        Returns:
            maximum B1+ in [T].
        """
        return float(np.abs(rf_envelope).max())
