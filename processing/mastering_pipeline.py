
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def ai_master(input_mix, output_master, target_lufs="-14"):
    run(f'''
    ffmpeg -i {input_mix} -af
    "equalizer=f=120:t=q:w=1:g=1,
     equalizer=f=10000:t=q:w=1:g=1,
     acompressor=threshold=-16dB:ratio=2,
     alimiter=limit=0.98,
     loudnorm=I={target_lufs}:TP=-1.0:LRA=11"
    {output_master}
    ''')
    return output_master
