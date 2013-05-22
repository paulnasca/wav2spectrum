#!/usr/bin/python

#
# Copyright (c) 2013 Paul Octavian Nasca, http://www.paulnasca.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# This is a simple program which analyses a small part of an input wav file 
# and outputs the spectrum as an audio sweep to a wav file 


import argparse
import sys
import scipy.io.wavfile
import scipy.signal
import scipy.ndimage
import numpy.fft
import numpy
import math

extra_silence=0.5

#load a wav file and return the data as mono values (-1.0,1.0) and the sample rate
def load_wav(filename):
    try:
        wavedata=scipy.io.wavfile.read(filename)
        samplerate=int(wavedata[0])
        smp=wavedata[1]*(1.0/32768.0)
        if len(smp.shape)>1: #convert to mono
            smp=(smp[:,0]+smp[:,1])*0.5
        return (samplerate,smp)
    except:
        print "Error loading wav: "+filename
        sys.exit(1)

#normalize the mono samples and save as a wav file
def save_wav(fname,samplerate,data):
	scipy.io.wavfile.write(fname,samplerate,numpy.int16(32767.0*(data/(1e-6+data.max()))))


#zoom the range of frequencies and interpolate the result
def zoom_and_interpolate(in_fft,min_freq_Hz,max_freq_Hz,samplerate,output_size):
    in_fft_len=len(in_fft)
    in_x=numpy.linspace(0,samplerate*0.5,in_fft_len)
    out_x=numpy.linspace(min_freq_Hz,max_freq_Hz,output_size)
    out_fft=numpy.interp(out_x,in_x,in_fft,0.0,0.0)
    return out_fft

#attempt to find the peaks in the data 
#not very accurate, but it gives an approximate values of the peaks 
def print_peaks(in_fft,min_freq_Hz,max_freq_Hz,samplerate):
    output_size=1000
    peak_width=20
    max_peaks=10

    out_fft=zoom_and_interpolate(in_fft,min_freq_Hz,max_freq_Hz,samplerate,output_size)
    out_fft_size=len(out_fft)
    
    out_fft_smooth_diff=out_fft-scipy.ndimage.gaussian_filter(out_fft,peak_width)
    out_fft_smooth_diff[out_fft_smooth_diff<0]=0.0

    peaks=[]

    for peak_k in range(0,int(output_size/peak_width)-1):
        pos1=peak_k*peak_width
        fft_slice=out_fft_smooth_diff[pos1:pos1+peak_width]
        max_index=numpy.argmax(fft_slice)+pos1
        max_freq_val=max_index/float(out_fft_size)*(max_freq_Hz-min_freq_Hz)+min_freq_Hz
        max_val=numpy.max(fft_slice)
        if (max_val>1e-4):
            peaks.append((max_val,max_freq_val))

    peaks=sorted(peaks,key=lambda peaks:peaks[0],reverse=True)
    peaks=peaks[:max_peaks]
    peaks=sorted(zip(*peaks)[1])
    print "Approximate peaks frequencies (Hz): ",
    for peak in peaks:
        print int(peak),
    print
    
    

#convert a small chunk from the input audio to an audio spectrum
#input_smp: input audio
#samplerate: sample rate
#window_size_seconds: the window(chunk) size from the input audio (seconds)
#input_position_seconds: the position of the window in the input audio (seconds)
#min_freq_Hz: the minimum frequency analysed from the input
#max_freq_Hz: the maximum frequency analysed from the input
#output_seconds: the length of the output audio
#limit_output_dB: the threshold (relative to maximum peak) of which frequencies are cut 
def get_audio_spectrum(input_smp, samplerate, window_size_seconds, input_position_seconds, min_freq_Hz, max_freq_Hz, output_seconds, limit_output_dB, print_peaks_enabled):

    #compute sizes
    window_size=max(16,int(window_size_seconds*samplerate))
    output_size=max(window_size*2+1,int(output_seconds*samplerate))
    input_position_samples=int(input_position_seconds*samplerate)
    input_size=len(input_smp)
   
    if window_size>=input_size:
        print "Window_size too large (it must be less than input audio size)"
        sys.exit(1)
    else:
        #compute the position from the input audio from which the window is extracted
        pos1=min(input_size-window_size,input_position_samples-window_size//2)
        pos1=max(0,pos1)

        #extract the window and normalize it
        sweep_in=input_smp[pos1:pos1+window_size]*(0.5-0.5*numpy.cos(numpy.linspace(0,2.0*numpy.pi,window_size)))
        sweep_in/=max(abs(sweep_in))+1e-6

   
    #analyse the sound
    in_fft=numpy.abs(numpy.fft.rfft(sweep_in))
    
    #print the peaks (if requested)
    if print_peaks_enabled:
        print_peaks(in_fft,min_freq_Hz,max_freq_Hz,samplerate)
   
    #cut off the frequencies which has too low amplitude
    limit_output=pow(10.0,limit_output_dB/20)*max(in_fft)
    in_fft[in_fft<limit_output]=0.0
    in_fft_len=len(in_fft)
    
    #generate the output spectrum which is scaled in order to fit the output sweeped sound
    out_fft=zoom_and_interpolate(in_fft,min_freq_Hz,max_freq_Hz,samplerate,output_size)

    #generate the sweep of a sine and multiply with the output spectrum 
    out_freqs=numpy.linspace(min_freq_Hz,max_freq_Hz,output_size)/samplerate
    xvalues=numpy.cumsum(out_freqs)*2.0*numpy.pi
    sweep_out=numpy.sin(xvalues)*out_fft

    sweep_out=sweep_out/(max(abs(sweep_out))+1e-6)


    #add noise to know the start/end of the sweeped output signal
    noise_amplitude=0.2
    sweep_out[0]=noise_amplitude
    sweep_out[1]=-noise_amplitude

    sweep_out[-20]=-noise_amplitude
    sweep_out[-21]=-noise_amplitude
    sweep_out[-2]=noise_amplitude
    sweep_out[-3]=noise_amplitude

    extra_silence_smp=numpy.zeros(int(samplerate*extra_silence))

    #the resulting sound is concatenation of: input_window + extra_silence + sweep_output signal + extra_silence
    sweep_out=numpy.concatenate((sweep_in,extra_silence_smp,sweep_out,extra_silence_smp))

    return sweep_out

def force_float_arguments_range(min_val,max_val):
    def do_force_float_arguments_range(string):
        v=float(string)
        if v<min_val or v>max_val:
            raise argparse.ArgumentTypeError("Value has to be between "+string(min_val)+" and "+string(max_val)+" .")
        return v
    return do_force_float_arguments_range

def force_float_arguments_positive_value(string):
    v=float(string)
    if v<0:
        raise argparse.ArgumentTypeError("Value has to be positive")
    return v

parser = argparse.ArgumentParser(description='Audio Spectrum by Paul Nasca ( http://www.paulnasca.com )', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i","--input",required=True, help="input WAV file")
parser.add_argument("-o","--output",required=True, help="output WAV file")
parser.add_argument("-p","--input_position",type=force_float_arguments_positive_value,default=0.0,help="window position (seconds) in the input audio")
parser.add_argument("-w","--window_size",type=force_float_arguments_positive_value, default=0.25,help="window size(seconds)")
parser.add_argument("-f","--min_freq",type=force_float_arguments_range(0.0,20000), default=20.0,help="minimum frequency (Hz)")
parser.add_argument("-F","--max_freq",type=force_float_arguments_range(0.0,20000), default=3000.0,help="maximum frequency (Hz)")
parser.add_argument("-z","--output_size",type=force_float_arguments_positive_value, default=10.0,help="output size (seconds)")
parser.add_argument("-l","--limit_output", type=force_float_arguments_range(-120,-6), default=-80.0,help="limit output (dB)")
parser.add_argument("-e","--print_peaks", action='store_true', default=False)
parser.add_argument("-t","--output_frequencies_positions", action='store_true', default=False)

if len(sys.argv)==1:
    parser.print_help()
    parser.parse_args() #show error message
    sys.exit(1)

arguments=parser.parse_args()

#input parameters
in_filename=arguments.input
out_filename=arguments.output
input_position_seconds=arguments.input_position
window_size_seconds=arguments.window_size
min_freq_Hz=arguments.min_freq
max_freq_Hz=arguments.max_freq
output_size_seconds=arguments.output_size
limit_output_dB=arguments.limit_output
print_peaks_enabled=arguments.print_peaks
output_frequencies_positions=arguments.output_frequencies_positions


tmp=load_wav(in_filename)
samplerate=tmp[0]
smp=numpy.float32(tmp[1])


smp_spectrum=get_audio_spectrum(smp,samplerate,window_size_seconds,input_position_seconds,min_freq_Hz,max_freq_Hz,output_size_seconds,limit_output_dB,print_peaks_enabled)

save_wav(out_filename,samplerate,smp_spectrum)

if output_frequencies_positions:
    output_size_ms=int(output_size_seconds*1000.0)
    start_ms=int((window_size_seconds+extra_silence)*1000.0)
    with open(out_filename+".txt", "a") as tf:
        tf.write("#time_milliseconds frequency_Hz\n")
        for i in range(output_size_ms):
            fx=float(i)/output_size_ms
            freq_Hz=min_freq_Hz*(1-fx)+max_freq_Hz*fx
            tf.write("%d %d\n" % (i+start_ms, int(freq_Hz)))





