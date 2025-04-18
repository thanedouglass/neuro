import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import re

# Create a dataframe to store the EEG power data
data = {
    'Subject': [],
    'Condition': [],
    'Delta_Power': [],
    'Theta_Power': [],
    'Alpha_Power': [],
    'Beta_Power': [],
    'Gamma_Power': []
}

eeg_output_data = """
----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 262.9590
Theta Power (4-8 Hz): 57.6565
Alpha Power (8-12 Hz): 36.4904
Beta Power (12-30 Hz): 19.2681
Gamma Power (30-100 Hz): 3.2214
Processing Eyes_Closed_Rashaun_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 288.7885
Theta Power (4-8 Hz): 96.5263
Alpha Power (8-12 Hz): 60.2019
Beta Power (12-30 Hz): 38.7767
Gamma Power (30-100 Hz): 5.3712
Processing Eyes_Closed_Haley_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 1380.8521
Theta Power (4-8 Hz): 279.6045
Alpha Power (8-12 Hz): 227.5173
Beta Power (12-30 Hz): 108.1153
Gamma Power (30-100 Hz): 15.8562
Processing Eyes_Closed_Tamia_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 186.2472
Theta Power (4-8 Hz): 112.9417
Alpha Power (8-12 Hz): 126.2225
Beta Power (12-30 Hz): 37.4697
Gamma Power (30-100 Hz): 7.4711
Processing Eyes_Closed_Thane_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 1055.3350
Theta Power (4-8 Hz): 407.8801
Alpha Power (8-12 Hz): 268.9006
Beta Power (12-30 Hz): 88.5137
Gamma Power (30-100 Hz): 11.1844
Processing Eyes_Open_Rashaun_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 462.4706
Theta Power (4-8 Hz): 192.2169
Alpha Power (8-12 Hz): 88.2130
Beta Power (12-30 Hz): 28.6544
Gamma Power (30-100 Hz): 3.5432
Processing Eyes_Open_Rhyan_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 198.0532
Theta Power (4-8 Hz): 66.7044
Alpha Power (8-12 Hz): 38.6194
Beta Power (12-30 Hz): 21.1764
Gamma Power (30-100 Hz): 3.5293
Processing Eyes_Open_Haley_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 555.5858
Theta Power (4-8 Hz): 143.6072
Alpha Power (8-12 Hz): 74.2768
Beta Power (12-30 Hz): 58.8339
Gamma Power (30-100 Hz): 14.2159
Processing Eyes_Open_Tamia_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 204.8264
Theta Power (4-8 Hz): 97.9981
Alpha Power (8-12 Hz): 70.1481
Beta Power (12-30 Hz): 33.3096
Gamma Power (30-100 Hz): 9.2440
Processing Eyes_Open_Thane_Drawing.wav (Frequency Domain)...

----- EEG Frequency Band Analysis -----
Delta Power (0.5-4 Hz): 302.0192
Theta Power (4-8 Hz): 118.1566
Alpha Power (8-12 Hz): 61.4029
Beta Power (12-30 Hz): 26.4726
Gamma Power (30-100 Hz): 4.9802
"""

sections = eeg_output_data.strip().split("----- EEG Frequency Band Analysis -----")
for section in sections:
    if not section.strip():
        continue
    
    # Extract power values
    delta_match = re.search(r"Delta Power \(0\.5-4 Hz\): (\d+\.\d+)", section)
    theta_match = re.search(r"Theta Power \(4-8 Hz\): (\d+\.\d+)", section)
    alpha_match = re.search(r"Alpha Power \(8-12 Hz\): (\d+\.\d+)", section)
    beta_match = re.search(r"Beta Power \(12-30 Hz\): (\d+\.\d+)", section)
    gamma_match = re.search(r"Gamma Power \(30-100 Hz\): (\d+\.\d+)", section)
    
    # Extract filename
    filename_match = re.search(r"Processing (.+)\.wav", section)
    
    if delta_match and theta_match and alpha_match and beta_match and gamma_match and filename_match:
        filename = filename_match.group(1)
        
        # Parse condition and subject from filename (Eyes_Closed_Rashaun_Drawing)
        parts = filename.split('_')
        if len(parts) >= 3:
            condition = f"{parts[0]}_{parts[1]}"
            subject = parts[2]
            
            # Add to data dictionary
            data['Subject'].append(subject)
            data['Condition'].append(condition)
            data['Delta_Power'].append(float(delta_match.group(1)))
            data['Theta_Power'].append(float(theta_match.group(1)))
            data['Alpha_Power'].append(float(alpha_match.group(1)))
            data['Beta_Power'].append(float(beta_match.group(1)))
            data['Gamma_Power'].append(float(gamma_match.group(1)))

# Convert to DataFrame
df = pd.DataFrame(data)

# Print summary statistics
print("Summary Statistics:")
print(df.groupby('Condition').describe())

# Prepare for long-format dataframe for easier plotting
df_long = pd.melt(
    df, 
    id_vars=['Subject', 'Condition'], 
    value_vars=['Delta_Power', 'Theta_Power', 'Alpha_Power', 'Beta_Power', 'Gamma_Power'],
    var_name='Band',
    value_name='Power'
)

# Remove _Power suffix from band names
df_long['Band'] = df_long['Band'].str.replace('_Power', '')

# Perform ANOVA tests for each frequency band
frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

print("\nANOVA Test Results:")
p_values = {}

for band in frequency_bands:
    # Get data for different conditions for this band
    eyes_closed = df[df['Condition'] == 'Eyes_Closed'][f'{band}_Power']
    eyes_open = df[df['Condition'] == 'Eyes_Open'][f'{band}_Power']
    
    if len(eyes_closed) > 0 and len(eyes_open) > 0:
        # Run ANOVA
        f_stat, p_val = stats.f_oneway(eyes_closed, eyes_open)
        p_values[band] = p_val
        
        print(f"\n{band} Band:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        
        alpha = 0.05
        if p_val < alpha:
            print(f"→ Significant difference between conditions (p < 0.05)")
        else:
            print(f"→ No significant difference between conditions (p >= 0.05)")

# Create boxplots with p-values displayed
plt.figure(figsize=(15, 10))

for i, band in enumerate(frequency_bands):
    plt.subplot(2, 3, i+1)

    # Add this dictionary definition before the plotting code:
    band_ranges = {
    'Delta': '0.5-4 Hz',
    'Theta': '4-8 Hz',
    'Alpha': '8-12 Hz',
    'Beta': '12-30 Hz',
    'Gamma': '30-100 Hz'
    }
    
    # Create boxplot for this band
    sns.boxplot(x='Condition', y=f'{band}_Power', data=df, palette='Set2')
    sns.stripplot(x='Condition', y=f'{band}_Power', data=df, color='black', alpha=0.5, jitter=True)
    
    # Add p-value to plot if available
    if band in p_values:
        p_val = p_values[band]
        sig_text = f"p = {p_val:.4f}"
        if p_val < 0.05:
            sig_text += " *"
        if p_val < 0.01:
            sig_text += "*"
        if p_val < 0.001:
            sig_text += "*"
            
        plt.text(0.5, 0.95, sig_text, transform=plt.gca().transAxes, 
                 horizontalalignment='center', fontsize=12)
    
    plt.title(f'{band} Power ({band} Band: {band_ranges[band]})')
    plt.ylabel('Power (µV²)')
    plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('eeg_band_power_analysis.png')
plt.show()

# Create a comprehensive boxplot for Alpha power (typically most relevant for EEG studies)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Condition', y='Alpha_Power', data=df, palette='Set2')
sns.stripplot(x='Condition', y='Alpha_Power', data=df, color='black', alpha=0.5, jitter=True)

# Add p-value
if 'Alpha' in p_values:
    p_val = p_values['Alpha']
    plt.text(0.5, 0.95, f"p = {p_val:.4f}", transform=plt.gca().transAxes, 
             horizontalalignment='center', fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.5))

plt.title('Alpha Power Comparison Between Conditions', fontsize=16)
plt.ylabel('Alpha Power (µV²)', fontsize=14)
plt.xlabel('Condition', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('alpha_power_comparison.png')
plt.show()

print("\nAnalysis complete. Results saved to eeg_band_power_analysis.png and alpha_power_comparison.png")
