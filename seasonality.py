# Tools to work with seasonality 

import numpy as np
import pandas as pd
import scipy.fftpack as fp
from scipy.signal import find_peaks
from typing import Dict, Tuple, List


class FFTDetector:
    """Modified FFT Detector based off of https://github.com/facebookresearch/Kats
    
    Fast Fourier Transform (FFT) Multiple Seasonality Detector

    Using FFT, detect the presence of seasonality and cycle length (e.g. yearly, monthly, etc.)

    Attributes:
        data: The input data (impact/trend data)
        label_dict: Dictionary of labels for each period length (in days)
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.label_dict = {
            "Annual": 365.0,
            "Biannual": 180.0,
            "Quarterly": 90.0,
            "Monthly": 0.0,
        }

    def label_seasonality(self):
        """Classify seasonality using Fourier transform decomposition.
        Returns:
            tuple: tuple of length 2 containing:
                1. seasonality_label ("seasonal" or "constant")
                2. if seasonal: seasonality_types; if constant: empty list
        """
        if len(self.data) == 0:
            return ("constant", [])

        try:
            self.data = self.data[["norm"]].reset_index()
        except KeyError:
            return ("constant", [])

        # Compute multiple seasonalities with FFT
        seasonality_results = self.detector()

        # Seasonality classification and multi-seasonality detection
        if seasonality_results["seasonality_presence"] == True:
            seasonalities = seasonality_results["seasonalities"]
            cycles = []
            for s in seasonalities:
                cycle = min(self.label_dict.items(), key=lambda x: abs(s - x[1]))[0]
                cycles.append(cycle)
            return ("seasonal", cycles)
        else:
            return ("constant", [])

    def detector(self, mad_threshold: float = 6.0) -> Dict:
        """Detect seasonality with FFT
        Args:
            mad_threshold: Optional; float; constant for the outlier algorithm for peak
                detector. The larger the value the less sensitive the outlier algorithm
                is.
        Returns:
            FFT Plot with peaks, selected peaks, and outlier boundary line.
        """
        fft = self.get_fft()
        _, orig_peaks, peaks = self.get_fft_peaks(fft, mad_threshold)

        seasonality_presence = len(peaks.index) > 0
        selected_seasonalities = []
        if seasonality_presence:
            selected_seasonalities = peaks["freq"].transform(lambda x: 1 / x).tolist()

        return {
            "seasonality_presence": seasonality_presence,
            "seasonalities": selected_seasonalities,
        }

    def get_fft(self) -> pd.DataFrame:
        """Computes FFT
        Returns:
            DataFrame with columns 'freq' and 'ampl'.
        """
        data_fft = fp.fft(self.data["norm"].values)
        data_psd = np.abs(data_fft) ** 2
        fftfreq = fp.fftfreq(len(data_psd), 1.0)
        pos_freq_ix = fftfreq > 0

        freq = (fftfreq[pos_freq_ix],)
        ampl = (10 * np.log10(data_psd[pos_freq_ix]),)

        return pd.DataFrame({"freq": freq[0], "ampl": ampl[0]})

    def get_fft_peaks(
        self, fft: pd.DataFrame, mad_threshold: float = 6.0
    ) -> Tuple[float, List[float], List[float]]:
        """Computes peaks in fft, selects the highest peaks (outliers) and
            removes the harmonics (multiplies of the base harmonics found)
        Args:
            fft: FFT computed by FFTDetector.get_fft
            mad_threshold: Optional; constant for the outlier algorithm for peak detector.
                The larger the value the less sensitive the outlier algorithm is.
        Returns:
            outlier threshold, peaks, selected peaks.
        """
        pos_fft = fft.loc[fft["ampl"] > 0]
        median = pos_fft["ampl"].median()
        pos_fft_above_med = pos_fft[pos_fft["ampl"] > median]
        mad = pos_fft_above_med["ampl"].mad()

        threshold = median + mad * mad_threshold

        peak_indices = find_peaks(fft["ampl"], threshold=0.1)
        peaks = fft.loc[peak_indices[0], :]

        orig_peaks = peaks.copy()

        peaks = peaks.loc[peaks["ampl"] > threshold].copy()
        peaks["Remove"] = [False] * len(peaks.index)
        peaks.reset_index(inplace=True)

        # Filter out harmonics
        for idx1 in range(len(peaks)):
            curr = peaks.loc[idx1, "freq"]
            for idx2 in range(idx1 + 1, len(peaks)):
                if peaks.loc[idx2, "Remove"] is True:
                    continue
                fraction = (peaks.loc[idx2, "freq"] / curr) % 1
                if fraction < 0.01 or fraction > 0.99:
                    peaks.loc[idx2, "Remove"] = True
        peaks = peaks.loc[~peaks["Remove"]]
        peaks.drop(inplace=True, columns="Remove")
        return threshold, orig_peaks, peaks