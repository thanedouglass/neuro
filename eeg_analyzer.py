import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from glob import glob
import argparse

# Set the specific path for your files
DEFAULT_DIRECTORY = "/Users/thanedouglass/Desktop/neurodrawing+"

def process_time_domain(file_path, save_dir=None):
    """
    Perform time-domain analysis on a single EEG file.
    
    Parameters:
    file_path (str): Path to the WAV file
    save_dir (str): Directory to save outputs (if None, will use file's directory)
    
    Returns:
    dict: Dictionary with processed data
    """
    filename = os.path.basename(file_path)
    print(f"Processing {filename} (Time Domain)...")
    
    try:
        # Read audio data
        audio_data, sample_rate = sf.read(file_path)
        
        # If stereo, handle it but keep both channels for visualization
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            print(f"Multi-channel data detected: {audio_data.shape[1]} channels")
            # Keep a mono version for spike detection
            mono_data = np.mean(audio_data, axis=1)
        else:
            mono_data = audio_data
        
        # Create visualization directory if saving
        if save_dir:
            vis_dir = os.path.join(save_dir, "Visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        else:
            vis_dir = os.path.join(os.path.dirname(file_path), "Visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        # Create time vector
        num_samp = len(audio_data)
        time_vector = np.linspace(0, num_samp / sample_rate, num_samp)
        
        # Visualize raw signal
        plt.figure(figsize=(12, 6))
        
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            for ch in range(audio_data.shape[1]):
                plt.plot(time_vector, audio_data[:, ch], label=f"Channel {ch+1}")
            plt.legend()
        else:
            plt.plot(time_vector, audio_data)
        
        plt.title(f"EEG Signal: {os.path.splitext(filename)[0]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        base_name = os.path.splitext(filename)[0]
        plt.savefig(os.path.join(vis_dir, f"{base_name}_time_domain.png"))
        plt.show()
        plt.close()
        
        # Detect spikes
        spike_indices = detect_spikes(mono_data)
        print(f"Detected {len(spike_indices)} spikes")
        
        if len(spike_indices) > 0:
            # Visualize spikes
            visualize_spikes(mono_data, sample_rate, spike_indices, time_vector, 
                            os.path.splitext(filename)[0], vis_dir)
            
            # Extract spike segments
            segments = extract_spike_segments(mono_data, sample_rate, spike_indices)
            
            # Save spike segments
            segments_dir = os.path.join(save_dir if save_dir else os.path.dirname(file_path), "Spike_Segments")
            save_spike_segments(segments, segments_dir, base_name, sample_rate)
        
        return {
            'filename': filename,
            'data': audio_data,
            'sample_rate': sample_rate,
            'spike_indices': spike_indices
        }
        
    except Exception as e:
        print(f"Error in time domain processing: {e}")
        return None

def process_frequency_domain(file_path, save_dir=None):
    """
    Perform frequency-domain analysis on a single EEG file.
    
    Parameters:
    file_path (str): Path to the WAV file
    save_dir (str): Directory to save outputs (if None, will use file's directory)
    
    Returns:
    dict: Dictionary with processed frequency data
    """
    filename = os.path.basename(file_path)
    print(f"Processing {filename} (Frequency Domain)...")
    
    try:
        # Read audio data
        data, sample_rate = sf.read(file_path)
        
        # If stereo, convert to mono for frequency analysis
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            print("Converting stereo to mono for frequency analysis")
        
        # Create visualization directory if saving
        if save_dir:
            vis_dir = os.path.join(save_dir, "Visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        else:
            vis_dir = os.path.join(os.path.dirname(file_path), "Visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        # Number of samples
        n = len(data)
        
        # Apply FFT
        fft_vals = np.fft.rfft(data)
        fft_freqs = np.fft.rfftfreq(n, 1 / sample_rate)
        
        # Power spectrum (magnitude)
        power = np.abs(fft_vals)
        
        # Define all common frequency bands
        delta_range = (fft_freqs >= 0.5) & (fft_freqs <= 4)
        theta_range = (fft_freqs >= 4) & (fft_freqs <= 8)
        alpha_range = (fft_freqs >= 8) & (fft_freqs <= 12)
        beta_range = (fft_freqs >= 12) & (fft_freqs <= 30)
        gamma_range = (fft_freqs >= 30) & (fft_freqs <= 100)
        
        # Compute average power in each band
        delta_power = np.mean(power[delta_range])
        theta_power = np.mean(power[theta_range])
        alpha_power = np.mean(power[alpha_range])
        beta_power = np.mean(power[beta_range])
        gamma_power = np.mean(power[gamma_range])
        
        # Show results
        print("\n----- EEG Frequency Band Analysis -----")
        print(f"Delta Power (0.5–4 Hz): {delta_power:.4f}")
        print(f"Theta Power (4–8 Hz): {theta_power:.4f}")
        print(f"Alpha Power (8–12 Hz): {alpha_power:.4f}")
        print(f"Beta Power (12–30 Hz): {beta_power:.4f}")
        print(f"Gamma Power (30–100 Hz): {gamma_power:.4f}")
        
        # Plot power spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freqs, power)
        plt.xlim(0, 50)  # Limit to 50Hz for better visualization
        plt.title(f"EEG Power Spectrum: {os.path.splitext(filename)[0]}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        
        # Add vertical bands for different frequency ranges
        plt.axvspan(0.5, 4, color='r', alpha=0.2, label='Delta')
        plt.axvspan(4, 8, color='g', alpha=0.2, label='Theta')
        plt.axvspan(8, 12, color='b', alpha=0.2, label='Alpha')
        plt.axvspan(12, 30, color='y', alpha=0.2, label='Beta')
        plt.axvspan(30, 50, color='m', alpha=0.2, label='Gamma')
        plt.legend()
        
        base_name = os.path.splitext(filename)[0]
        plt.savefig(os.path.join(vis_dir, f"{base_name}_freq_domain.png"))
        plt.show()
        plt.close()
        
        # Create bar chart of power in different bands
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        powers = [delta_power, theta_power, alpha_power, beta_power, gamma_power]
        
        plt.figure(figsize=(10, 6))
        plt.bar(bands, powers, color=['r', 'g', 'b', 'y', 'm'])
        plt.title(f"EEG Frequency Band Power: {os.path.splitext(filename)[0]}")
        plt.xlabel("Frequency Band")
        plt.ylabel("Average Power")
        plt.savefig(os.path.join(vis_dir, f"{base_name}_band_power.png"))
        plt.show()
        plt.close()
        
        return {
            'filename': filename,
            'delta_power': delta_power,
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'beta_power': beta_power,
            'gamma_power': gamma_power,
            'fft_freqs': fft_freqs,
            'power': power
        }
        
    except Exception as e:
        print(f"Error in frequency domain processing: {e}")
        return None

def detect_spikes(data, threshold_factor=3.0):
    """
    Simple spike detection based on amplitude threshold.
    
    Parameters:
    data (array): EEG data
    threshold_factor (float): Factor multiplied by standard deviation to set threshold
    
    Returns:
    array: Indices where spikes occur
    """
    # Calculate threshold based on data standard deviation
    threshold = threshold_factor * np.std(data)
    
    # Find indices where absolute value exceeds threshold
    spike_indices = np.where(np.abs(data) > threshold)[0]
    
    # Group adjacent indices (spikes within 10 samples of each other)
    if len(spike_indices) > 0:
        spike_groups = [[spike_indices[0]]]
        for i in range(1, len(spike_indices)):
            if spike_indices[i] - spike_indices[i-1] <= 10:
                spike_groups[-1].append(spike_indices[i])
            else:
                spike_groups.append([spike_indices[i]])
        
        # Take the middle index of each group as the spike location
        spike_locations = [int(np.mean(group)) for group in spike_groups]
        return np.array(spike_locations)
    
    return np.array([])

def visualize_spikes(data, sample_rate, spike_indices, time_vector=None, title=None, save_dir=None):
    """
    Visualize detected spikes in the EEG data.
    
    Parameters:
    data (array): EEG data
    sample_rate (int): Sample rate of the recording
    spike_indices (array): Indices where spikes were detected
    time_vector (array): Pre-computed time vector (optional)
    title (str): Title for the plot
    save_dir (str): Directory to save the visualization
    """
    # Create time vector if not provided
    if time_vector is None:
        num_samp = len(data)
        time_vector = np.linspace(0, num_samp / sample_rate, num_samp)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the EEG data
    plt.plot(time_vector, data, 'b')
    
    # Plot detected spikes
    plt.plot(time_vector[spike_indices], data[spike_indices], 'ro', label='Detected Spikes')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"EEG with Detected Spikes: {title}" if title else "EEG with Detected Spikes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir and title:
        plt.savefig(os.path.join(save_dir, f"{title}_spikes.png"))
    
    plt.show()
    plt.close()

def extract_spike_segments(data, sample_rate, spike_indices, window_ms=100):
    """
    Extract segments around detected spikes.
    
    Parameters:
    data (array): EEG data
    sample_rate (int): Sample rate of the recording
    spike_indices (array): Indices where spikes were detected
    window_ms (int): Window size in milliseconds (±window_ms around spike)
    
    Returns:
    list: List of spike segments
    """
    # Calculate window size in samples
    window_samples = int((window_ms / 1000) * sample_rate)
    
    segments = []
    for spike_idx in spike_indices:
        # Calculate segment boundaries
        start_idx = max(0, spike_idx - window_samples)
        end_idx = min(len(data), spike_idx + window_samples)
        
        # Extract segment
        segment = data[start_idx:end_idx]
        segments.append(segment)
    
    return segments

def save_spike_segments(segments, output_folder, base_filename, sample_rate):
    """
    Save spike segments to WAV files.
    
    Parameters:
    segments (list): List of spike segments
    output_folder (str): Folder to save segments
    base_filename (str): Base name for the segment files
    sample_rate (int): Sample rate for the output WAV files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_folder, f"{base_filename}_spike_{i+1}.wav")
        sf.write(output_file, segment, sample_rate)
        print(f"Saved spike segment to {output_file}")

def main():
    """Main function to run the EEG analysis tool"""
    
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='EEG Analysis Tool')
    parser.add_argument('-m', '--mode', type=str, default='prompt',
                        choices=['time', 'freq', 'both', 'prompt'],
                        help='Analysis mode: time domain, frequency domain, both, or prompt')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='Specific EEG WAV file to analyze')
    parser.add_argument('-d', '--dir', type=str, default=DEFAULT_DIRECTORY,
                        help='Directory containing EEG WAV files')
    
    args = parser.parse_args()
    mode = args.mode
    file_path = args.file
    directory = args.dir
    
    # If no directory or file specified, use the default directory
    if not directory and not file_path:
        directory = DEFAULT_DIRECTORY
    
    # If mode is 'prompt', ask the user what they want to do
    if mode == 'prompt':
        print("\n*** EEG Analysis Tool ***\n")
        print("Choose analysis mode:")
        print("a) Time Domain Processing (signal visualization, spike detection)")
        print("b) Frequency Domain Analysis (alpha/beta/delta/theta/gamma power calculation)")
        print("c) Both Time and Frequency Domain Analysis")
        
        choice = input("\nEnter choice (a/b/c): ").strip().lower()
        
        if choice == 'a':
            mode = 'time'
        elif choice == 'b':
            mode = 'freq'
        elif choice == 'c':
            mode = 'both'
        else:
            print("Invalid choice. Defaulting to both.")
            mode = 'both'
    
    # Process a single file if specified
    if file_path:
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return
            
        if mode == 'time' or mode == 'both':
            process_time_domain(file_path, os.path.dirname(file_path))
        
        if mode == 'freq' or mode == 'both':
            process_frequency_domain(file_path, os.path.dirname(file_path))
    
    # Process all .wav files in the directory
    elif directory:
        if not os.path.exists(directory):
            print(f"Directory {directory} not found.")
            return
            
        wav_files = glob(os.path.join(directory, "*.wav"))
        
        if not wav_files:
            print(f"No WAV files found in {directory}")
            return
            
        print(f"Found {len(wav_files)} WAV files in {directory}")
        
        for wav_file in wav_files:
            if mode == 'time' or mode == 'both':
                process_time_domain(wav_file, directory)
            
            if mode == 'freq' or mode == 'both':
                process_frequency_domain(wav_file, directory)
    
if __name__ == "__main__":
    main()
