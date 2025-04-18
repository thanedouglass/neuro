# Investigating Brain Activity During Multitasking

This repository contains the complete dataset and analysis tools for the NeuroDrawing+ project, which investigates the effects of multitasking (specifically drawing while listening to music) on brain activity, with a focus on resting state alpha and beta waves. Our research explores how visual processing and attention are affected during creative activities when performed simultaneously with auditory processing tasks.

**Research Question**
:How does multitasking (drawing + music listening) affect EEG brainwave patterns, particularly in the alpha (8-12 Hz) and beta (12-30 Hz) frequency ranges, compared to single-task conditions?

**Methodology**
:We collected EEG recordings from participants in two conditions:

1. **Eyes Closed**: Resting state with eyes closed (baseline)
2. **Eyes Open**: Active drawing while listening to music

Data was collected using a consumer-grade EEG headset with recordings saved as WAV files for analysis. Multiple participants (Rashaun, Hailey, Tamia, Thane, and Rhyan) contributed to the dataset, providing a diverse sample for analysis.

**Processing Pipeline**

1. **Signal Acquisition**: EEG data recorded as WAV files
2. **Time Domain Analysis**: Signal visualization and spike detection
3. **Frequency Domain Analysis**: Fast Fourier Transform (FFT) to extract power across frequency bands
4. **Statistical Analysis**: ANOVA tests to compare conditions across frequency bands

**Key Findings**
:Statistical analysis revealed no significant differences between the Eyes Closed and Eyes Open conditions across all frequency bands (Delta, Theta, Alpha, Beta, and Gamma). Though the differences weren't statistically significant, we observed interesting variations in the mean power across conditions, particularly in the alpha band where Eyes Closed showed a trend toward higher power (as is typically expected in resting state).
→ Delta Band: F = 0.0126, p = 0.9137
→ Theta Band: F = 0.3094, p = 0.5954
→ Alpha Band: F = 0.0059, p = 0.9411
→ Beta Band: F = 0.0470, p = 0.8345
→ Gamma Band: F = 0.0113, p = 0.9182 
The p-values for all bands were above the significance threshold (p ≥ 0.05):

**How to Reproduce This Study**

Prerequisites: 
→ Python3+
→ Consumer-grade EEG headset (we used BYB Human SpikerBox + DVR-esque interface)
→ Drawing materials

**Required Python Packages**
```
pip install numpy matplotlib seaborn scipy pandas soundfile
```
Setup recording environment:
→ Quiet room with minimal distractions
→ Comfortable seating for participants
→ Drawing materials ready


Record EEG data:
→ Baseline (Eyes Closed): 2-minute recording with eyes closed
→ Experimental (Eyes Open): 2-minute recording while drawing and listening to music
→ Save recordings as WAV files with naming convention

**Run Analysis**
```
# Process all recordings
python scripts/eeg_analyzer.py --mode freq --dir ./data

# Generate statistics and visualizations
python scripts/eeg_analysis.py
```


This project is licensed under the MIT License - see the LICENSE file for details.

